# SPDX-License-Identifier: Apache-2.0
"""Fish Speech S2 Pro engine — standalone Dual-AR TTS pipeline.

Pipeline:
    1. Tokenize text + build prompt (with optional voice clone conditioning)
    2. Slow AR (Qwen3 backbone) → semantic token generation
    3. Fast AR (4-layer residual predictor) → residual codebook codes
    4. DAC decoder → 44.1 kHz waveform

This engine is standalone — it loads the HuggingFace model directly and runs
autoregressive generation without ATOM's batched scheduler, because Fish Speech
requires custom inputs_embeds with codebook conditioning.

Requires the ``fish-speech`` package for DAC codec architecture.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from atom.models.fish_speech.configuration_fish_speech import FishSpeechConfig
from atom.models.fish_speech.dac_utils import DAC_SAMPLE_RATE, build_dac_codec
from atom.models.fish_speech.prompt_utils import (
    build_fish_text_only_prompt_ids,
    build_fish_voice_clone_prompt_ids,
)

logger = logging.getLogger("atom.fish_speech")


class FishSpeechEngine:
    """Fish Speech S2 Pro standalone engine.

    Loads the full model (Slow AR + Fast AR + DAC codec) and runs end-to-end
    text-to-speech synthesis.
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
        self._codec = None
        self._config = None
        self._sample_rate = DAC_SAMPLE_RATE  # 44100 Hz

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def model_type(self) -> str:
        return "fish_speech"

    def load(self) -> None:
        """Load all components: model, tokenizer, and DAC codec."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        t0 = time.time()

        # Load config
        config = AutoConfig.from_pretrained(str(self.model_dir), trust_remote_code=True)
        self._config = config

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), trust_remote_code=True,
        )

        # Load model
        logger.info("Loading Fish Speech S2 Pro from %s on %s (%s)",
                     self.model_dir, self.device, self.dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            torch_dtype=self.dtype,
            device_map=str(self.device),
            trust_remote_code=True,
        )
        self._model.eval()

        # Load DAC codec for decoding
        self._load_codec()

        elapsed = time.time() - t0
        params_m = sum(p.numel() for p in self._model.parameters()) / 1e6
        logger.info(
            "Fish Speech loaded: %.0fM params, %.1fs, VRAM=%.0fMB",
            params_m, elapsed,
            torch.cuda.memory_allocated(self.device) / 1e6 if self.device.type == "cuda" else 0,
        )

    def _load_codec(self) -> None:
        """Load DAC codec for waveform decoding."""
        codec_path = self.model_dir / "codec.pth"
        if not codec_path.exists():
            try:
                from transformers.utils.hub import cached_file
                cached = cached_file(str(self.model_dir), "codec.pth")
                if cached:
                    codec_path = Path(cached)
            except ImportError:
                pass

        if not codec_path.exists():
            logger.warning("codec.pth not found — DAC decode will be unavailable")
            return

        codec = build_dac_codec()
        state_dict = torch.load(str(codec_path), map_location="cpu", weights_only=True)
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        codec.load_state_dict(state_dict, strict=False)
        codec = codec.to(device=self.device, dtype=torch.float32)
        codec.eval()
        self._codec = codec
        logger.info("Loaded DAC codec from %s", codec_path)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        ref_audio_path: str | None = None,
        ref_text: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.2,
        seed: int | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict]:
        """Generate speech audio from text.

        Returns:
            (wav_array, metrics_dict) — 1-D float32 audio at 44.1kHz and timing metrics.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        metrics = {}

        # 1. Build prompt
        t0 = time.time()
        if ref_audio_path and ref_text:
            input_ids = self._build_voice_clone_prompt(text, ref_audio_path, ref_text)
        else:
            input_ids = self._build_text_prompt(text)
        metrics["prompt_sec"] = time.time() - t0
        metrics["prompt_len"] = len(input_ids)

        # 2. Generate semantic tokens
        t1 = time.time()
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Get semantic token range from config
        config = self._config
        if hasattr(config, "semantic_start_token_id"):
            semantic_start = config.semantic_start_token_id
            semantic_end = config.semantic_end_token_id
        elif hasattr(config, "text_config"):
            semantic_start = config.text_config.semantic_begin_id
            semantic_end = config.text_config.semantic_end_id
        else:
            semantic_start = 151678
            semantic_end = 155773

        # im_end token for stopping
        im_end_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")

        output = self._model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            eos_token_id=im_end_id,
        )
        generated = output[0, len(input_ids):]
        metrics["generate_sec"] = time.time() - t1

        # Extract semantic tokens (filter to valid range)
        semantic_mask = (generated >= semantic_start) & (generated <= semantic_end)
        semantic_tokens = generated[semantic_mask]
        # Convert to codebook indices
        semantic_codes = (semantic_tokens - semantic_start).cpu()
        metrics["num_tokens"] = len(semantic_codes)
        metrics["tok_per_sec"] = metrics["num_tokens"] / max(metrics["generate_sec"], 0.001)

        if len(semantic_codes) == 0:
            logger.warning("No semantic tokens generated")
            return np.zeros(1, dtype=np.float32), metrics

        # 3. Decode to audio via DAC
        t2 = time.time()
        if self._codec is not None:
            wav = self._decode_dac(semantic_codes)
        else:
            logger.warning("DAC codec not loaded — returning silence")
            wav = np.zeros(int(self._sample_rate * 0.1), dtype=np.float32)
        metrics["decode_sec"] = time.time() - t2
        metrics["audio_duration"] = len(wav) / self._sample_rate
        metrics["total_sec"] = sum(metrics[k] for k in ["prompt_sec", "generate_sec", "decode_sec"])
        metrics["rtf"] = metrics["total_sec"] / max(metrics["audio_duration"], 0.001)

        return wav, metrics

    def _build_text_prompt(self, text: str) -> list[int]:
        """Build text-only prompt for Fish Speech."""
        prompt_ids, _ = build_fish_text_only_prompt_ids(self._tokenizer, text)
        return prompt_ids

    def _build_voice_clone_prompt(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
    ) -> list[int]:
        """Build voice clone prompt with DAC-encoded reference audio."""
        from atom.models.fish_speech.dac_encoder import encode_reference_audio_codes

        # Load and encode reference audio
        if ref_audio_path.startswith("data:"):
            # Base64 data URL
            import base64
            _, data = ref_audio_path.split(",", 1)
            import io
            audio_bytes = base64.b64decode(data)
            wav, sr = sf.read(io.BytesIO(audio_bytes))
        else:
            wav, sr = sf.read(ref_audio_path)

        ref_codes = encode_reference_audio_codes(
            str(self.model_dir), wav, sr, device=self.device,
        )

        # Build semantic token IDs from codebook 0
        config = self._config
        if hasattr(config, "semantic_start_token_id"):
            semantic_start = config.semantic_start_token_id
        elif hasattr(config, "text_config"):
            semantic_start = config.text_config.semantic_begin_id
        else:
            semantic_start = 151678

        semantic_token_ids = (ref_codes[:, 0] + semantic_start).tolist()

        prompt_ids, _, _ = build_fish_voice_clone_prompt_ids(
            self._tokenizer, text, ref_text, semantic_token_ids,
        )
        return prompt_ids

    def _decode_dac(self, semantic_codes: torch.Tensor) -> np.ndarray:
        """Decode semantic codes to waveform using DAC codec.

        For standalone mode without Fast AR, we use only the semantic codebook
        (codebook 0) and zero-fill residual codebooks.
        """
        num_codebooks = 10  # DAC uses 10 codebooks
        num_frames = len(semantic_codes)

        # Build full codebook tensor: [1, num_codebooks, num_frames]
        codes = torch.zeros(1, num_codebooks, num_frames, dtype=torch.long, device=self.device)
        codes[0, 0, :] = semantic_codes.to(self.device)

        # Decode
        with torch.no_grad():
            wav = self._codec.decode(codes)

        # [1, 1, samples] → 1-D numpy
        wav_np = wav.squeeze().cpu().float().numpy()
        return wav_np

    def list_voices(self) -> list[str]:
        """Fish Speech has no built-in voices — voice cloning only."""
        return []
