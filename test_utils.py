import sys
from unittest.mock import MagicMock
sys.modules['aiter'] = MagicMock()
sys.modules['aiter.dist'] = MagicMock()
sys.modules['aiter.dist.shm_broadcast'] = MagicMock()
sys.modules['aiter.utility'] = MagicMock()
sys.modules['aiter.utility.dtypes'] = MagicMock()
sys.modules['vllm'] = MagicMock()

import torch
import torchaudio
import numpy as np

from atom.audio.utils import apply_speed_adjustment

# 1D audio test
audio = np.random.randn(24000).astype(np.float32)
audio_adj, sr = apply_speed_adjustment(audio, 1.5, 24000)
print("Original shape:", audio.shape)
print("Adjusted shape:", audio_adj.shape)
