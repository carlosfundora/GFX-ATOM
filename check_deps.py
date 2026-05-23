import sys
import numpy as np
import time

try:
    import rs_codec
    print("rs_codec found!")
except ImportError as e:
    print(f"rs_codec error: {e}")

try:
    import soundfile as sf
    print(f"soundfile found! {sf.__version__}")
except ImportError as e:
    print(f"soundfile error: {e}")

try:
    import soxr
    print(f"soxr found! {soxr.__version__}")
except ImportError as e:
    print(f"soxr error: {e}")
