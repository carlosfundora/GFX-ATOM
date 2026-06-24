import sys
from unittest.mock import MagicMock
sys.modules['aiter'] = MagicMock()
sys.modules['aiter.dist'] = MagicMock()
sys.modules['aiter.dist.shm_broadcast'] = MagicMock()
sys.modules['aiter.utility'] = MagicMock()
sys.modules['aiter.utility.dtypes'] = MagicMock()
sys.modules['vllm'] = MagicMock()

import torch

import rs_codec
print("rs_codec imported successfully")

from atom.audio.chatterbox.engine import ChatterboxEngine

try:
    engine = ChatterboxEngine(model_dir="dummy", device="cpu", dtype="float32")
    print("Engine initialized")
except Exception as e:
    print(f"Engine init failed: {e}")
