import sys
from unittest.mock import MagicMock
sys.modules['aiter'] = MagicMock()
sys.modules['aiter.dist'] = MagicMock()
sys.modules['aiter.dist.shm_broadcast'] = MagicMock()
sys.modules['aiter.utility'] = MagicMock()
sys.modules['aiter.utility.dtypes'] = MagicMock()
sys.modules['vllm'] = MagicMock()

import torch

from atom.audio.chatterbox.engine import ChatterboxEngine

# Just testing it still imports fine
engine = ChatterboxEngine(model_dir="dummy", device="cpu", dtype="float32")
print("Engine initialized")
