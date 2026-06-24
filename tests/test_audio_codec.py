# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for the unified audio codec module.

Validates:
- Raw PCM decode/encode
- Base64 audio decode/encode
- Resampling with multiple backends
- RMS computation
- Format conversions
- Error handling and fallback behavior
"""

import base64
import io
import pytest
import numpy as np

from atom.audio.codec import (
    AudioCodec,
    AudioDecodeError,
    AudioEncodeError,
    MIN_AUDIO_DURATION_S,
    MAX_AUDIO_DURATION_S,
    SAMPLE_RATE_INPUT,
    SAMPLE_RATE_OUTPUT,
)


class TestAudioCodecBasics:
    """Test basic codec functionality."""

    def test_codec_backend_info(self):
        """Get information about available backends."""
        info = AudioCodec.get_backend_info()
        
        assert isinstance(info, dict)
        assert "soundfile" in info
        assert "torchaudio" in info
        assert "soxr" in info
        assert "rs_codec" in info

    def test_as_audio_f32_conversion(self):
        """Ensure audio conversion to float32."""
        audio_int16 = np.array([0, 1000, -1000], dtype=np.int16)
        audio_f32 = AudioCodec._as_audio_f32(audio_int16)
        
        assert audio_f32.dtype == np.float32
        assert audio_f32[0] == 0

    def test_empty_audio_handling(self):
        """Handle empty audio gracefully."""
        empty = np.array([], dtype=np.float32)
        
        rms = AudioCodec.compute_rms(empty)
        assert rms == 0.0


class TestAudioCodecDecoding:
    """Test audio decoding functionality."""

    def test_decode_raw_int16(self):
        """Decode raw int16 PCM bytes."""
        # Create simple int16 data: [0, 1000, -1000]
        data = np.array([0, 1000, -1000], dtype=np.int16)
        raw_bytes = data.tobytes()
        
        decoded = AudioCodec.decode_raw(raw_bytes, source_sr=16000, target_sr=16000)
        
        assert decoded.dtype == np.float32
        assert len(decoded) == 3
        assert abs(decoded[1] - 1000/32768.0) < 0.01
        assert abs(decoded[2] - (-1000/32768.0)) < 0.01

    def test_decode_raw_float32(self):
        """Decode raw float32 PCM bytes."""
        data = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        raw_bytes = data.tobytes()
        
        decoded = AudioCodec.decode_raw(raw_bytes, source_sr=16000, target_sr=16000, dtype="float32")
        
        assert decoded.dtype == np.float32
        np.testing.assert_array_almost_equal(decoded, data)

    def test_decode_raw_empty(self):
        """Handle empty raw bytes."""
        decoded = AudioCodec.decode_raw(b"", source_sr=16000, target_sr=16000)
        
        assert len(decoded) == 0
        assert decoded.dtype == np.float32

    def test_decode_b64_empty_with_validation(self):
        """Empty base64 raises with validation enabled."""
        with pytest.raises(ValueError, match="Empty audio"):
            AudioCodec.decode_b64("", validate=True)

    def test_decode_b64_empty_without_validation(self):
        """Empty base64 returns empty array without validation."""
        audio, sr = AudioCodec.decode_b64("", validate=False)
        
        assert len(audio) == 0
        assert audio.dtype == np.float32

    def test_decode_b64_invalid_base64(self):
        """Invalid base64 raises error."""
        with pytest.raises(AudioDecodeError, match="Invalid base64"):
            AudioCodec.decode_b64("not!!!valid!!!base64", validate=True)

    def test_decode_audio_convenience(self):
        """Convenience method decode_audio."""
        # Create audio long enough to pass validation (> 100ms = 1600 samples at 16kHz)
        data = np.array([0] + [16384, -16384] * 1000, dtype=np.int16)
        raw_bytes = data.tobytes()
        audio_b64 = base64.b64encode(raw_bytes).decode("utf-8")
        
        # Decode at 16kHz (no resample needed)
        audio, sr = AudioCodec.decode_audio(audio_b64, sample_rate=16000)
        
        assert sr == SAMPLE_RATE_INPUT
        assert audio.dtype == np.float32
        assert len(audio) >= 1600  # At least 100ms worth


class TestAudioCodecEncoding:
    """Test audio encoding functionality."""

    def test_encode_pcm_bytes(self):
        """Encode float32 to PCM16 bytes."""
        audio = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        
        pcm_bytes = AudioCodec.encode_pcm_bytes(audio)
        
        assert isinstance(pcm_bytes, bytes)
        assert len(pcm_bytes) == len(audio) * 2  # 16-bit = 2 bytes per sample
        
        # Verify we can decode it back
        decoded = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(decoded, audio, decimal=3)

    def test_encode_wav_b64(self):
        """Encode float32 to base64 WAV string."""
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        
        wav_b64 = AudioCodec.encode_wav_b64(audio, sample_rate=16000)
        
        assert isinstance(wav_b64, str)
        
        # Verify it's valid base64
        wav_bytes = base64.b64decode(wav_b64)
        assert wav_bytes.startswith(b"RIFF")

    def test_encode_wav_b64_roundtrip(self):
        """Encode and decode WAV with roundtrip verification."""
        original_audio = np.array([0.0, 0.25, 0.5, 0.75, -0.5], dtype=np.float32)
        
        # Encode
        wav_b64 = AudioCodec.encode_wav_b64(original_audio, sample_rate=16000)
        
        # Decode
        decoded_audio, sr = AudioCodec.decode_b64(wav_b64, source_sr=16000, target_sr=16000, validate=False)
        
        assert sr == 16000
        assert len(decoded_audio) == len(original_audio)
        # Allow some numerical error due to compression
        np.testing.assert_array_almost_equal(decoded_audio, original_audio, decimal=3)


class TestAudioCodecResampling:
    """Test audio resampling functionality."""

    def test_resample_no_op(self):
        """No resample when source and target rate are same."""
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        
        resampled = AudioCodec.resample(audio, from_sr=16000, to_sr=16000)
        
        np.testing.assert_array_equal(resampled, audio)

    def test_resample_int_to_f32(self):
        """Resample converts dtype to float32."""
        audio = np.array([0, 1000, -1000], dtype=np.int16)
        
        # The codec should auto-convert to f32 for resampling
        resampled = AudioCodec.resample(
            AudioCodec._as_audio_f32(audio),  # Pre-convert to f32 as codec expects
            from_sr=16000,
            to_sr=16000
        )
        
        assert resampled.dtype == np.float32

    def test_resample_downsampling(self):
        """Downsample from 44.1kHz to 16kHz."""
        # Create 1 second at 44.1kHz
        sr_orig = 44100
        duration_sec = 0.5
        samples = int(sr_orig * duration_sec)
        audio = np.sin(2 * np.pi * 440 * np.arange(samples) / sr_orig).astype(np.float32)
        
        resampled = AudioCodec.resample(audio, from_sr=sr_orig, to_sr=16000)
        
        # Check output length is approximately correct
        expected_length = int(samples * 16000 / sr_orig)
        assert abs(len(resampled) - expected_length) <= 2

    def test_resample_upsampling(self):
        """Upsample from 8kHz to 16kHz."""
        sr_orig = 8000
        duration_sec = 0.5
        samples = int(sr_orig * duration_sec)
        audio = np.sin(2 * np.pi * 440 * np.arange(samples) / sr_orig).astype(np.float32)
        
        resampled = AudioCodec.resample(audio, from_sr=sr_orig, to_sr=16000)
        
        # Output should be roughly 2x length
        expected_length = int(samples * 2)
        assert abs(len(resampled) - expected_length) <= 4

    def test_resample_empty_audio(self):
        """Resample handles empty audio."""
        empty = np.array([], dtype=np.float32)
        
        resampled = AudioCodec.resample(empty, from_sr=16000, to_sr=8000)
        
        assert len(resampled) == 0

    def test_resample_single_sample(self):
        """Resample single sample audio."""
        audio = np.array([0.5], dtype=np.float32)
        
        resampled = AudioCodec.resample(audio, from_sr=16000, to_sr=8000)
        
        assert len(resampled) > 0
        assert resampled.dtype == np.float32


class TestAudioCodecAnalysis:
    """Test audio analysis functions."""

    def test_compute_rms_silent(self):
        """RMS of silent audio is near zero."""
        silent = np.zeros(16000, dtype=np.float32)
        
        rms = AudioCodec.compute_rms(silent)
        
        assert rms < 0.01

    def test_compute_rms_tone(self):
        """RMS of sine wave is predictable."""
        # 1 second sine wave at 1kHz, 16kHz sample rate
        t = np.arange(16000) / 16000
        audio = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        
        rms = AudioCodec.compute_rms(audio)
        
        # RMS of sine wave at amplitude 0.5 is 0.5/sqrt(2) ≈ 0.354
        assert 0.3 < rms < 0.4

    def test_compute_rms_int_input(self):
        """RMS works with int16 input."""
        audio = np.array([0, 1000, -1000, 500, -500], dtype=np.int16)
        
        rms = AudioCodec.compute_rms(audio)
        
        assert rms > 0

    def test_float_to_int16_bytes(self):
        """Convert float32 frame to int16 bytes."""
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        
        int16_bytes = AudioCodec.float_to_int16_bytes(audio)
        
        assert isinstance(int16_bytes, bytes)
        assert len(int16_bytes) == len(audio) * 2
        
        # Verify roundtrip
        recovered = np.frombuffer(int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(recovered, audio, decimal=3)


class TestAudioCodecValidation:
    """Test validation constraints."""

    def test_decode_b64_validates_duration_min(self):
        """Audio shorter than minimum raises error."""
        # Create very short audio (10ms at 16kHz = 160 samples)
        too_short = np.array(np.random.randn(160), dtype=np.float32) * 0.1
        
        wav_b64 = AudioCodec.encode_wav_b64(too_short, sample_rate=16000)
        
        with pytest.raises(ValueError, match="too short"):
            AudioCodec.decode_b64(wav_b64, validate=True)

    def test_decode_b64_validates_duration_max(self):
        """Audio longer than maximum raises error."""
        # Create very long audio (400 seconds at 16kHz)
        too_long = np.array(np.random.randn(400 * 16000), dtype=np.float32) * 0.1
        
        wav_b64 = AudioCodec.encode_wav_b64(too_long, sample_rate=16000)
        
        with pytest.raises(ValueError, match="too long"):
            AudioCodec.decode_b64(wav_b64, validate=True)

    def test_decode_b64_skips_validation_when_disabled(self):
        """Very short audio allowed when validation disabled."""
        too_short = np.array(np.random.randn(160), dtype=np.float32) * 0.1
        
        wav_b64 = AudioCodec.encode_wav_b64(too_short, sample_rate=16000)
        
        # Should not raise with validate=False
        audio, sr = AudioCodec.decode_b64(wav_b64, validate=False)
        assert len(audio) > 0


class TestAudioCodecEdgeCases:
    """Test edge cases and error handling."""

    def test_decode_raw_odd_byte_count(self):
        """Handle odd byte count gracefully."""
        # Create int16 data but corrupt it to odd byte count
        data = np.array([0, 1000, -1000], dtype=np.int16)
        raw_bytes = data.tobytes()[:-1]  # Remove last byte
        
        # Should still decode (truncates extra byte)
        decoded = AudioCodec.decode_raw(raw_bytes, source_sr=16000, target_sr=16000)
        
        assert len(decoded) == 2  # Lost one sample

    def test_float_to_int16_clipping(self):
        """Extreme float values are clipped to valid int16 range."""
        audio = np.array([0.0, 10.0, -10.0], dtype=np.float32)  # Extreme values
        
        int16_bytes = AudioCodec.float_to_int16_bytes(audio)
        
        # Should not raise, and values should be clipped
        recovered = np.frombuffer(int16_bytes, dtype=np.int16)
        assert np.max(np.abs(recovered)) <= 32767

    def test_encode_wav_b64_invalid_input_shape(self):
        """Handle multi-dimensional audio gracefully."""
        # 2D audio should be squeezed or handled
        audio_2d = np.array([[0.0, 0.5], [-0.5, 1.0]], dtype=np.float32)
        
        # Should handle 2D input
        wav_b64 = AudioCodec.encode_wav_b64(audio_2d.flatten(), sample_rate=16000)
        assert isinstance(wav_b64, str)
