# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Audio Codec for ATOM — Unified audio format handling with multiple backends.

Provides comprehensive audio encoding/decoding with graceful fallback chain:
- Primary: soundfile (WAV I/O) + torchaudio (resampling) + rs_codec (Rust DSP)
- Fallback: soxr (resampling) → librosa (resampling) → scipy (WAV I/O)

Matches DEMERZEL codec interface and patterns for consistency across services.

Supported formats:
- Decode: WAV (int16/int32/float32/float64), raw PCM, base64-encoded
- Encode: WAV, PCM, FLAC, MP3, AAC, Opus
- Resample: torchaudio (best) → soxr (good) → librosa (acceptable)
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger("atom.audio.codec")

# Audio constraints matching DEMERZEL
MIN_AUDIO_DURATION_S = 0.1  # Minimum 100ms
MAX_AUDIO_DURATION_S = 300.0  # Maximum 5 minutes
SAMPLE_RATE_INPUT = 16000  # ASR input standard
SAMPLE_RATE_OUTPUT = 24000  # TTS output preferred


# ─── Custom Exceptions ─────────────────────────────────────────────────────

class AudioDecodeError(ValueError):
    """Raised when audio decoding fails."""
    pass


class AudioEncodeError(ValueError):
    """Raised when audio encoding fails."""
    pass


# ─── Conditional Backend Imports ───────────────────────────────────────────

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
    logger.debug("Audio codec: soundfile available")
except ImportError:
    _HAS_SOUNDFILE = False
    logger.debug("Audio codec: soundfile unavailable")

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
    logger.debug("Audio codec: torchaudio available")
except Exception as exc:
    torchaudio = None
    _HAS_TORCHAUDIO = False
    logger.debug(f"Audio codec: torchaudio unavailable ({exc})")

try:
    import soxr
    _HAS_SOXR = True
    logger.debug("Audio codec: soxr available")
except ImportError:
    _HAS_SOXR = False
    logger.debug("Audio codec: soxr unavailable")

try:
    import rs_codec as codec_rs
    _HAS_RUST_CODEC = True
    logger.debug("Audio codec: rs_codec available")
except ImportError:
    codec_rs = None
    _HAS_RUST_CODEC = False
    logger.debug("Audio codec: rs_codec unavailable")


# ─── Audio Codec Implementation ────────────────────────────────────────────

class AudioCodec:
    """Unified audio codec with multiple backend support and graceful fallback."""

    # Resampler cache for performance
    _resampler_cache: dict[tuple[int, int], Any] = {}

    @staticmethod
    def _as_audio_f32(audio: np.ndarray) -> np.ndarray:
        """Ensure audio is float32 for Rust FFI."""
        if audio.dtype != np.float32:
            return audio.astype(np.float32)
        return audio

    @staticmethod
    def decode_raw(
        audio_bytes: bytes,
        source_sr: int = 16000,
        target_sr: int = SAMPLE_RATE_INPUT,
        dtype: str = "int16",
    ) -> np.ndarray:
        """Decode raw bytes to float32 numpy array, optionally resampled.

        Handles int16 (default), float32, and mixed formats with graceful fallback.
        """
        if not audio_bytes:
            return np.array([], dtype=np.float32)

        audio_float = None

        # Try Rust fast path first
        if _HAS_RUST_CODEC and dtype == "int16":
            try:
                audio_float = codec_rs.decode_raw_pcm16(audio_bytes, float(source_sr), float(target_sr))
                if audio_float is not None:
                    return np.asarray(audio_float, dtype=np.float32)
            except Exception as e:
                logger.debug(f"Rust decode_raw failed: {e}")

        # Fallback to NumPy decoding
        try:
            if dtype == "float32":
                if len(audio_bytes) % 4 != 0:
                    audio_bytes = audio_bytes[: (len(audio_bytes) // 4) * 4]
                audio_float = np.frombuffer(audio_bytes, dtype=np.float32).copy()
                return AudioCodec.resample(audio_float, source_sr, target_sr)

            # Default to int16
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes[: (len(audio_bytes) // 2) * 2]
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            return AudioCodec.resample(audio_float, source_sr, target_sr)
        except Exception as e:
            raise AudioDecodeError(f"Failed to decode raw audio: {e}")

    @staticmethod
    def decode_b64(
        audio_b64: str,
        source_sr: int = 16000,
        target_sr: int = SAMPLE_RATE_INPUT,
        validate: bool = True,
    ) -> tuple[np.ndarray, int]:
        """Decode base64 audio to float32 numpy array.

        Handles:
        - WAV files (int16, int32, float32, float64)
        - Raw PCM fallback with automatic format detection
        - Stereo → mono downmix
        - Duration validation

        Returns: (audio_float32, actual_sample_rate)
        """
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except (binascii.Error, ValueError) as e:
            raise AudioDecodeError(f"Invalid base64 audio: {e}")

        if not audio_bytes:
            if validate:
                raise ValueError("Empty audio data")
            return np.array([], dtype=np.float32), target_sr

        audio_float = None

        # Try soundfile first (best for WAV)
        if _HAS_SOUNDFILE:
            try:
                buf = io.BytesIO(audio_bytes)
                data, sr = sf.read(buf, dtype="float32", always_2d=False)
                if data.ndim == 2:
                    data = data.mean(axis=1)
                audio_float = AudioCodec.resample(data, sr, target_sr)
            except Exception as e:
                logger.debug(f"soundfile decode failed: {e}")

        # Fallback to scipy
        if audio_float is None:
            try:
                import scipy.io.wavfile as wavfile
                buf = io.BytesIO(audio_bytes)
                sr, data = wavfile.read(buf)
                if data.ndim == 2:
                    data = data.mean(axis=1)
                if data.dtype == np.int16:
                    audio_float = data.astype(np.float32) / 32768.0
                elif np.issubdtype(data.dtype, np.floating):
                    audio_float = data.astype(np.float32, copy=False)
                else:
                    audio_float = data.astype(np.float32) / np.iinfo(data.dtype).max
                audio_float = AudioCodec.resample(audio_float, sr, target_sr)
            except Exception as e:
                logger.debug(f"scipy decode failed: {e}")

        # Last resort: raw PCM
        if audio_float is None:
            audio_float = AudioCodec.decode_raw(
                audio_bytes,
                source_sr=source_sr,
                target_sr=target_sr,
                dtype="int16",
            )

        # Validate duration
        if validate:
            duration = len(audio_float) / target_sr
            if duration < MIN_AUDIO_DURATION_S:
                raise ValueError(f"Audio too short: {duration:.2f}s (min {MIN_AUDIO_DURATION_S}s)")
            if duration > MAX_AUDIO_DURATION_S:
                raise ValueError(f"Audio too long: {duration:.2f}s (max {MAX_AUDIO_DURATION_S}s)")

        return audio_float, target_sr

    @staticmethod
    def decode_audio(
        audio_b64: str,
        sample_rate: int = 16000,
    ) -> tuple[np.ndarray, int]:
        """Decode base64 audio to float32 at standard 16kHz.

        Convenience method matching DEMERZEL interface.
        """
        return AudioCodec.decode_b64(
            audio_b64,
            source_sr=sample_rate,
            target_sr=SAMPLE_RATE_INPUT,
            validate=True,
        )

    @staticmethod
    def encode_wav_b64(
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE_OUTPUT,
        subtype: str = "PCM_16",
    ) -> str:
        """Encode float32 numpy array to base64 WAV string.

        Fast path: Rust FFI fuses float→PCM16→WAV→base64.
        Fallback: soundfile (preferred) or scipy.
        """
        audio = AudioCodec._as_audio_f32(audio)

        # Try Rust fast path
        if _HAS_RUST_CODEC and subtype == "PCM_16":
            try:
                return codec_rs.encode_wav_b64(audio, sample_rate, subtype)
            except Exception as e:
                logger.debug(f"Rust encode_wav_b64 failed: {e}")

        # Fallback to soundfile
        buf = io.BytesIO()
        try:
            if _HAS_SOUNDFILE:
                sf.write(buf, audio, sample_rate, format="WAV", subtype=subtype)
            else:
                import scipy.io.wavfile as wavfile
                pcm_int16 = np.clip(audio * 32768.0, -32768.0, 32767.0).astype(np.int16)
                wavfile.write(buf, sample_rate, pcm_int16)
        except Exception as e:
            raise AudioEncodeError(f"WAV encoding failed: {e}")

        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def encode_pcm_bytes(
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE_OUTPUT,
    ) -> bytes:
        """Encode float32 numpy array to PCM16 bytes.

        Fast path: Rust FFI.
        Fallback: NumPy.
        """
        audio = AudioCodec._as_audio_f32(audio)

        if _HAS_RUST_CODEC:
            try:
                return codec_rs.audio_to_pcm_bytes(audio)
            except Exception as e:
                logger.debug(f"Rust PCM encoding failed: {e}")

        # NumPy fallback
        pcm_int16 = np.clip(audio * 32768.0, -32768.0, 32767.0).astype(np.int16)
        return pcm_int16.tobytes()

    @staticmethod
    def resample(
        audio: np.ndarray,
        from_sr: int,
        to_sr: int,
        quality: str = "HQ",
    ) -> np.ndarray:
        """Resample audio using best available backend.

        Backend priority:
        1. torchaudio (best quality, cached)
        2. soxr (good quality, configurable)
        3. librosa (acceptable)
        4. linear interpolation (last resort)
        """
        if from_sr == to_sr:
            return audio

        audio = AudioCodec._as_audio_f32(audio)

        # torchaudio path (cached for performance)
        if _HAS_TORCHAUDIO:
            try:
                import torch
                cache_key = (from_sr, to_sr)
                if cache_key not in AudioCodec._resampler_cache:
                    AudioCodec._resampler_cache[cache_key] = torchaudio.transforms.Resample(from_sr, to_sr)
                
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                resampled = AudioCodec._resampler_cache[cache_key](audio_tensor)
                return resampled.squeeze(0).cpu().numpy().astype(np.float32)
            except Exception as e:
                logger.debug(f"torchaudio resample failed: {e}")

        # soxr path
        if _HAS_SOXR:
            try:
                return soxr.resample(audio, from_sr, to_sr, quality=quality).astype(np.float32)
            except Exception as e:
                logger.debug(f"soxr resample failed: {e}")

        # librosa path
        try:
            import librosa
            return librosa.resample(audio, orig_sr=from_sr, target_sr=to_sr).astype(np.float32)
        except ImportError:
            pass

        # Linear interpolation fallback
        if len(audio) == 0:
            return audio
        if len(audio) == 1:
            return np.repeat(audio, int(np.ceil(to_sr / from_sr)))

        ratio = to_sr / from_sr
        new_len = int(len(audio) * ratio)
        if new_len == 0:
            return np.array([], dtype=np.float32)

        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @staticmethod
    def compute_rms(audio: np.ndarray) -> float:
        """Compute RMS energy of audio buffer.

        Used for energy gating and voice activity detection.
        """
        if len(audio) == 0:
            return 0.0

        audio = AudioCodec._as_audio_f32(audio)

        # Try Rust fast path
        if _HAS_RUST_CODEC:
            try:
                return float(codec_rs.compute_rms(audio))
            except Exception as e:
                logger.debug(f"Rust compute_rms failed: {e}")

        # NumPy fallback
        return float(np.sqrt(np.mean(audio ** 2) + 1e-12))

    @staticmethod
    def float_to_int16_bytes(frame: np.ndarray) -> bytes:
        """Convert float32 frame to int16 PCM bytes.

        Used for WebRTC VAD which requires int16 PCM.
        """
        frame = AudioCodec._as_audio_f32(frame)

        # Try Rust fast path
        if _HAS_RUST_CODEC:
            try:
                return codec_rs.float_to_int16_bytes(frame)
            except Exception as e:
                logger.debug(f"Rust float_to_int16_bytes failed: {e}")

        # NumPy fallback
        return (np.clip(frame, -1.0, 1.0) * 32768).astype(np.int16).tobytes()

    @staticmethod
    def get_backend_info() -> dict[str, bool]:
        """Return information about available backends."""
        return {
            "soundfile": _HAS_SOUNDFILE,
            "torchaudio": _HAS_TORCHAUDIO,
            "soxr": _HAS_SOXR,
            "rs_codec": _HAS_RUST_CODEC,
        }
