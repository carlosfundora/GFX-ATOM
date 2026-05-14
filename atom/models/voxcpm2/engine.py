# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 engine — standalone TTS pipeline with voice cloning.

Pipeline:
    1. Text tokenization + prompt construction
    2. MiniCPM4 base_lm → FSQ → residual_lm → LocDiT → AudioVAE
    3. 48kHz waveform output

This engine loads the native VoxCPM2 model via the ``voxcpm`` package
and runs end-to-end synthesis. Voice cloning is supported via reference
audio (ref_audio) with optional transcript (ref_text).

Requires the ``voxcpm`` package (>= 2.0).
"""

import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger("atom.voxcpm2")

VOXCPM2_SAMPLE_RATE = 48000


def _import_voxcpm():
    """Import VoxCPM core, checking env var and sibling paths."""
    env_path = os.environ.get("ATOM_VOXCPM_CODE_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    try:
        from voxcpm.core import VoxCPM
        return VoxCPM
    except ImportError:
        raise ImportError(
            "Could not import voxcpm. Install with: uv pip install voxcpm>=2.0 "
            "Or set ATOM_VOXCPM_CODE_PATH to the VoxCPM source tree."
        )


def is_cjk_char(c: str) -> bool:
    """Check if a character is a CJK ideograph."""
    cp = ord(c)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2F800 <= cp <= 0x2FA1F
    )


def build_cjk_split_map(tokenizer: Any) -> dict[int, list[int]]:
    """Build {multichar_cjk_token_id: [single_char_ids]} from tokenizer vocab."""
    vocab = tokenizer.get_vocab()
    split_map: dict[int, list[int]] = {}
    for token, token_id in vocab.items():
        clean = token.replace("\u2581", "")
        if len(clean) >= 2 and all(is_cjk_char(c) for c in clean):
            char_ids = tokenizer.convert_tokens_to_ids(list(clean))
            if all(cid != tokenizer.unk_token_id for cid in char_ids):
                split_map[token_id] = char_ids
    return split_map


def split_multichar_chinese(token_ids: list[int], split_map: dict[int, list[int]]) -> list[int]:
    """Replace multichar Chinese token IDs with single-char IDs (idempotent)."""
    result: list[int] = []
    for tid in token_ids:
        expansion = split_map.get(tid)
        if expansion is not None:
            result.extend(expansion)
        else:
            result.append(tid)
    return result


class VoxCPM2Engine:
    """VoxCPM2 standalone TTS engine.

    Loads the native VoxCPM2 model and runs end-to-end synthesis with
    support for zero-shot TTS, voice cloning (ref_audio), and voice
    continuation (ref_audio + ref_text).
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self._model = None
        self._tokenizer = None
        self._split_map: dict[int, list[int]] = {}
        self._sample_rate = VOXCPM2_SAMPLE_RATE

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def model_type(self) -> str:
        return "voxcpm2"

    def load(self) -> None:
        """Load VoxCPM2 model and tokenizer."""
        t0 = time.time()

        VoxCPM = _import_voxcpm()
        logger.info("Loading VoxCPM2 from %s on %s (%s)",
                     self.model_dir, self.device, self.dtype)

        self._model = VoxCPM.from_pretrained(
            str(self.model_dir),
            load_denoiser=True,
            optimize=True,
        )
        self._model = self._model.to(self.device)
        self._model.eval()

        # Get tokenizer from model
        if hasattr(self._model, "tts_model") and hasattr(self._model.tts_model, "text_tokenizer"):
            tts = self._model.tts_model
            self._tokenizer = tts.text_tokenizer.tokenizer
            self._split_map = build_cjk_split_map(self._tokenizer)
            logger.info("Built CJK split map: %d entries", len(self._split_map))

        # Get sample rate from config
        if hasattr(self._model, "config"):
            cfg = self._model.config
            self._sample_rate = getattr(cfg, "sample_rate", VOXCPM2_SAMPLE_RATE)

        elapsed = time.time() - t0
        logger.info("VoxCPM2 loaded in %.1fs", elapsed)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        ref_audio_path: str | None = None,
        ref_text: str | None = None,
        voice: str | None = None,
        max_tokens: int = 2000,
        seed: int | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict]:
        """Generate speech audio from text.

        Args:
            text: Input text to synthesize.
            ref_audio_path: Reference audio for voice cloning.
            ref_text: Transcript of reference audio (for continuation mode).
            voice: Speaker preset name (if model supports it).
            max_tokens: Maximum decode steps.
            seed: Random seed for reproducibility.

        Returns:
            (wav_array, metrics_dict) — 1-D float32 audio at 48kHz.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        metrics = {}
        t0 = time.time()

        tts = self._model.tts_model if hasattr(self._model, "tts_model") else self._model

        # Build prompt cache for voice cloning
        prompt_cache = None
        if ref_audio_path:
            t_ref = time.time()
            prompt_cache = self._build_prompt_cache(
                tts, ref_audio_path, ref_text,
            )
            metrics["ref_encode_sec"] = time.time() - t_ref

        # Generate
        t_gen = time.time()
        try:
            if hasattr(self._model, "generate"):
                # Use native generate API
                wav_tensor = self._model.generate(
                    text=text,
                    prompt_cache=prompt_cache,
                    max_new_tokens=max_tokens,
                )
            elif hasattr(tts, "generate"):
                wav_tensor = tts.generate(
                    text=text,
                    prompt_cache=prompt_cache,
                    max_new_tokens=max_tokens,
                )
            else:
                raise RuntimeError("VoxCPM2 model has no generate method")
        except Exception as e:
            logger.error("VoxCPM2 generation failed: %s", e, exc_info=True)
            raise

        metrics["generate_sec"] = time.time() - t_gen

        # Convert to numpy
        if isinstance(wav_tensor, torch.Tensor):
            wav = wav_tensor.squeeze().cpu().float().numpy()
        else:
            wav = np.asarray(wav_tensor, dtype=np.float32).squeeze()

        metrics["audio_duration"] = len(wav) / self._sample_rate
        metrics["total_sec"] = time.time() - t0
        metrics["rtf"] = metrics["total_sec"] / max(metrics["audio_duration"], 0.001)
        metrics["num_tokens"] = int(metrics.get("audio_duration", 0) * 50)  # Estimate
        metrics["tok_per_sec"] = metrics["num_tokens"] / max(metrics["generate_sec"], 0.001)

        return wav, metrics

    def _build_prompt_cache(
        self,
        tts: Any,
        ref_audio_path: str,
        ref_text: str | None,
    ) -> dict | None:
        """Build prompt cache for voice cloning."""
        # Handle base64 data URLs
        if ref_audio_path.startswith("data:"):
            import base64
            import io
            _, data = ref_audio_path.split(",", 1)
            audio_bytes = base64.b64decode(data)
            wav_np, sr = sf.read(io.BytesIO(audio_bytes))
            samples = wav_np.tolist()

            if ref_text:
                return tts.build_prompt_cache(
                    prompt_text=ref_text,
                    prompt_wav_path=None,
                    reference_wav_path=None,
                )
            # Encode via AudioVAE
            audio = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
            encode_sr = getattr(tts, "_encode_sample_rate", sr)
            if sr != encode_sr:
                try:
                    import torchaudio
                    resampler = torchaudio.transforms.Resample(sr, encode_sr)
                    audio = resampler(audio)
                except ImportError:
                    pass
            feat = tts.audio_vae.encode(audio.to(self.device), encode_sr).cpu()
            patch_size = getattr(tts, "patch_size", 4)
            feat_dim = getattr(tts, "feat_dim", 64)
            feat = feat.view(feat_dim, -1, patch_size).permute(1, 2, 0)
            return {"ref_audio_feat": feat, "mode": "reference"}

        # File path
        if ref_text:
            return tts.build_prompt_cache(
                prompt_text=ref_text,
                prompt_wav_path=ref_audio_path,
            )
        return tts.build_prompt_cache(
            reference_wav_path=ref_audio_path,
        )

    def list_voices(self) -> list[str]:
        """VoxCPM2 supports zero-shot — no built-in presets."""
        return ["default"]
