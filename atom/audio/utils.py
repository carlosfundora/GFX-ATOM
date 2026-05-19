# SPDX-License-Identifier: Apache-2.0
"""Audio format conversion and processing utilities.

Supports WAV, PCM, FLAC, MP3, AAC, and Opus output formats with optional
speed adjustment via torchaudio phase vocoder.
"""

import io
import logging
import struct
from typing import Literal

import numpy as np
import soundfile as sf

try:
    import rs_codec
    _HAS_RS_CODEC = True
except ImportError:
    _HAS_RS_CODEC = False

logger = logging.getLogger("atom.audio")


def create_wav_header(
    sample_rate: int,
    num_channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Create a WAV header with placeholder size for streaming.

    Uses 0xFFFFFFFF as placeholder for data size fields, which is accepted
    by most audio clients and matches OpenAI's streaming WAV implementation.
    """
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    placeholder_size = 0xFFFFFFFF

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        placeholder_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        placeholder_size,
    )
    return header


def audio_to_bytes(
    audio: np.ndarray,
    sample_rate: int,
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav",
) -> tuple[bytes, str]:
    """Convert numpy audio array to bytes in the specified format.

    Returns (audio_bytes, media_type).
    """
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Ensure 1-D
    if audio.ndim > 1:
        audio = audio.squeeze()

    format_map = {
        "wav": ("WAV", "audio/wav", {"subtype": "PCM_16"}),
        "pcm": ("RAW", "audio/pcm", {"subtype": "PCM_16"}),
        "flac": ("FLAC", "audio/flac", {}),
        "mp3": ("MP3", "audio/mpeg", {}),
        "aac": ("AAC", "audio/aac", {}),
        "opus": ("OGG", "audio/ogg", {"subtype": "OPUS"}),
    }

    if response_format not in format_map:
        raise ValueError(f"Unsupported format: {response_format}")

    sf_format, media_type, kwargs = format_map[response_format]

    buf = io.BytesIO()
    if response_format == "pcm":
        # Raw 16-bit PCM (Optimized via Rust)
        if _HAS_RS_CODEC:
            pcm_bytes = rs_codec.audio_to_pcm_bytes(audio)
            buf.write(pcm_bytes)
        else:
            pcm_data = np.clip(audio * 32767, -32768, 32767).astype(np.int16, copy=False)
            buf.write(pcm_data.tobytes())
    else:
        sf.write(buf, audio, sample_rate, format=sf_format, **kwargs)

    return buf.getvalue(), media_type


def apply_speed_adjustment(
    audio: np.ndarray,
    speed: float,
    sample_rate: int,
) -> tuple[np.ndarray, int]:
    """Apply speed adjustment to audio while preserving pitch.

    Uses torchaudio's phase vocoder (Spectrogram -> TimeStretch ->
    InverseSpectrogram) to stretch/compress audio in time without
    changing pitch.

    Returns (adjusted_audio, sample_rate).
    """
    if speed == 1.0:
        return audio, sample_rate

    try:
        import torch
        import torchaudio
    except ImportError:
        logger.warning(
            "torchaudio not available for speed adjustment; returning original audio"
        )
        return audio, sample_rate

    try:
        with torch.inference_mode():
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32, copy=False)

            # Stereo numpy arrays use channels-last (T, C);
            # torch expects channels-first (C, T).
            channels_last = audio.ndim == 2
            if channels_last:
                waveform = torch.from_numpy(audio.T)
            else:
                waveform = torch.from_numpy(audio).unsqueeze(0)

            n_fft = 2048
            hop_length = n_fft // 4
            to_spec = torchaudio.transforms.Spectrogram(
                n_fft=n_fft, hop_length=hop_length, power=None,
            )
            stretch = torchaudio.transforms.TimeStretch(
                n_freq=n_fft // 2 + 1, hop_length=hop_length,
            )
            to_wave = torchaudio.transforms.InverseSpectrogram(
                n_fft=n_fft, hop_length=hop_length,
            )

            spec = to_spec(waveform)
            stretched = stretch(spec, speed)
            expected_length = int(audio.shape[0] / speed)
            result = to_wave(stretched, length=expected_length)

            result = result.squeeze(0).numpy()
            if channels_last:
                result = result.T
            return result, sample_rate
    except Exception as e:
        logger.error("Speed adjustment failed: %s", e)
        raise ValueError("Failed to apply speed adjustment.") from e
