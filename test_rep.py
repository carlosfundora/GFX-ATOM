import sys
from unittest.mock import MagicMock
sys.modules['aiter'] = MagicMock()
sys.modules['aiter.dist'] = MagicMock()
sys.modules['aiter.dist.shm_broadcast'] = MagicMock()
sys.modules['aiter.utility'] = MagicMock()
sys.modules['aiter.utility.dtypes'] = MagicMock()
sys.modules['vllm'] = MagicMock()

import numpy as np
import torch
from atom.audio.chatterbox.engine import ChatterboxEngine, RepetitionPenaltyProcessor

def test_numpy_rep_penalty():
    print("Testing np rep penalty")
    scores = np.array([[1.0, -1.0, 2.0, 3.0]])
    input_ids = np.array([[0, 1]])
    penalty = 1.2

    returned_scores = ChatterboxEngine._np_rep_penalty(input_ids, scores, penalty)

    assert returned_scores is scores, "Did not return original tensor"
    assert np.allclose(scores[0, 0], 1.0 / penalty)
    assert np.allclose(scores[0, 1], -1.0 * penalty)
    print("np rep penalty passed.")

def test_torch_rep_penalty():
    print("Testing torch rep penalty")
    scores = torch.tensor([[1.0, -1.0, 2.0, 3.0]], dtype=torch.float32)
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    penalty = 1.2
    processor = RepetitionPenaltyProcessor(penalty=penalty)

    returned_scores = processor(input_ids, scores)

    assert returned_scores is scores, "Did not return original tensor"
    assert torch.allclose(scores[0, 0], torch.tensor(1.0 / penalty))
    assert torch.allclose(scores[0, 1], torch.tensor(-1.0 * penalty))
    print("torch rep penalty passed.")

test_numpy_rep_penalty()
test_torch_rep_penalty()
