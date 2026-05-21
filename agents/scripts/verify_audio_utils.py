import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from atom.audio.utils import audio_to_bytes
import numpy as np

audio = np.array([0.5, -0.5, 1.1, -1.1], dtype=np.float32)
bytes_out, mime = audio_to_bytes(audio, 24000, "pcm")
assert mime == "audio/pcm"
expected = np.array([16383, -16383, 32767, -32768], dtype=np.int16).tobytes()
# There might be some rounding differences:
# 0.5 * 32767 = 16383.5 -> cast to int16 is 16383
assert bytes_out == expected
print("audio_to_bytes fallback verified successfully.")
