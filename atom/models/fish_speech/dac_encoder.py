# SPDX-License-Identifier: Apache-2.0
"""DAC codec encoder for Fish Speech S2 Pro voice cloning.

Encodes reference audio into VQ codes for use as prompt conditioning.

Ported from vLLM-Omni.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

from atom.models.fish_speech.dac_utils import DAC_SAMPLE_RATE, build_dac_codec

logger = logging.getLogger("atom.fish_speech")

_codec_cache: dict[tuple[str, str, str], nn.Module] = {}


def _load_dac_codec(
    model_path: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Load the DAC codec model from codec.pth."""
    device = torch.device(device)
    cache_key = (model_path, str(device), str(dtype))
    if cache_key in _codec_cache:
        return _codec_cache[cache_key]

    codec_path = os.path.join(model_path, "codec.pth")
    if not os.path.exists(codec_path):
        try:
            from transformers.utils.hub import cached_file
            cached = cached_file(model_path, "codec.pth")
            if cached is not None:
                codec_path = cached
        except ImportError:
            pass

    if not os.path.exists(codec_path):
        raise FileNotFoundError(
            f"codec.pth not found for {model_path}. "
            "Required for voice cloning with Fish Speech S2 Pro."
        )

    codec = build_dac_codec()
    state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
    codec.load_state_dict(state_dict, strict=False)
    # Encoder path only uses encoder + quantizer; prune the decoder.
    codec.decoder = None
    codec = codec.to(device=device, dtype=dtype)
    codec.eval()

    _codec_cache[cache_key] = codec
    logger.info("Loaded DAC codec encoder from %s (%s, dtype=%s)", codec_path, device, dtype)
    return codec


@lru_cache(maxsize=16)
def _get_resample_kernel(
    source_sr: int,
    target_sr: int,
    device: torch.device,
    dtype: torch.dtype,
):
    import torchaudio
    return torchaudio.transforms.Resample(source_sr, target_sr).to(device=device, dtype=dtype)


def _prepare_reference_audio_tensor(
    wav_samples: list[float] | np.ndarray | torch.Tensor,
    sample_rate: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(wav_samples, torch.Tensor):
        wav_tensor = wav_samples.detach()
    else:
        wav_tensor = torch.as_tensor(wav_samples)

    wav_tensor = wav_tensor.to(device=device, dtype=dtype)
    if wav_tensor.ndim == 2:
        if wav_tensor.shape[0] <= 8 and wav_tensor.shape[1] > wav_tensor.shape[0]:
            wav_tensor = wav_tensor.mean(dim=0)
        elif wav_tensor.shape[-1] <= 8 and wav_tensor.shape[0] > wav_tensor.shape[-1]:
            wav_tensor = wav_tensor.mean(dim=-1)
        else:
            wav_tensor = wav_tensor.mean(dim=0)
    elif wav_tensor.ndim > 2:
        wav_tensor = wav_tensor.reshape(-1, wav_tensor.shape[-1]).mean(dim=0)
    wav_tensor = wav_tensor.flatten()

    if sample_rate != DAC_SAMPLE_RATE:
        resampler = _get_resample_kernel(int(sample_rate), DAC_SAMPLE_RATE, device, dtype)
        wav_tensor = resampler(wav_tensor.unsqueeze(0)).squeeze(0)
    return wav_tensor


@torch.no_grad()
def encode_reference_audio_codes(
    model_path: str,
    wav_samples: list[float] | np.ndarray | torch.Tensor,
    sample_rate: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode reference audio into DAC codebook indices.

    Returns:
        Tensor of shape [num_frames, num_codebooks] on the requested device
        (dtype=torch.long).
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device)
    dtype = torch.float32
    codec = _load_dac_codec(model_path, device=device, dtype=dtype)
    wav_tensor = _prepare_reference_audio_tensor(
        wav_samples, sample_rate, device=device, dtype=dtype,
    )

    wav_tensor = wav_tensor.unsqueeze(0).unsqueeze(0)
    feature_lengths = torch.tensor([wav_tensor.shape[-1]], device=device, dtype=torch.long)
    codes, _ = codec.encode(wav_tensor, feature_lengths)

    # [1, num_codebooks, num_frames] -> [num_frames, num_codebooks]
    codes_fq = codes[0].transpose(0, 1).to(dtype=torch.long).contiguous()
    logger.info(
        "Encoded reference audio: %d samples @ %dHz -> frames=%d codebooks=%d",
        int(wav_tensor.shape[-1]), sample_rate,
        int(codes_fq.shape[0]), int(codes_fq.shape[1]),
    )
    return codes_fq
