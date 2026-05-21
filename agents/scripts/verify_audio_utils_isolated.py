import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import io
import struct
from typing import Literal

def audio_to_bytes(
    audio: np.ndarray,
    sample_rate: int,
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav",
) -> tuple[bytes, str]:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Ensure 1-D
    if audio.ndim > 1:
        audio = audio.squeeze()

    format_map = {
        "wav": ("WAV", "audio/wav", {"subtype": "PCM_16"}),
        "pcm": ("RAW", "audio/pcm", {"subtype": "PCM_16"}),
    }

    sf_format, media_type, kwargs = format_map[response_format]

    if response_format == "pcm":
        pcm_data = np.empty_like(audio, dtype=np.int16)
        np.clip(audio * 32767, -32768, 32767, out=pcm_data, casting='unsafe')
        return pcm_data.tobytes(), media_type
    else:
        raise ValueError("Only PCM in this test")

audio = np.array([0.5, -0.5, 1.1, -1.1], dtype=np.float32)
bytes_out, mime = audio_to_bytes(audio, 24000, "pcm")
assert mime == "audio/pcm"
expected = np.array([16383, -16383, 32767, -32768], dtype=np.int16).tobytes()
assert bytes_out == expected
print("audio_to_bytes fallback verified successfully.")
