import sys
from unittest.mock import MagicMock

sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torch.distributed'] = MagicMock()
sys.modules['torch.library'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock()

import numpy as np

import atom.audio.chatterbox.engine
ChatterboxEngine = atom.audio.chatterbox.engine.ChatterboxEngine

def test_engine_rep_penalty():
    vocab_size = 32000
    batch_size = 1
    seq_len = 512
    np.random.seed(42)
    scores = np.random.randn(batch_size, vocab_size).astype(np.float32)
    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(np.int64)
    penalty = 1.2

    # Check it runs without error
    out = ChatterboxEngine._np_rep_penalty(input_ids, scores.copy(), penalty)
    assert out.shape == scores.shape

