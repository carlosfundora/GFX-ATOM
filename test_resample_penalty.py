import torch
import numpy as np
import soxr

import sys
from unittest.mock import MagicMock
sys.modules['aiter'] = MagicMock()
sys.modules['aiter.dist'] = MagicMock()
sys.modules['aiter.dist.shm_broadcast'] = MagicMock()
sys.modules['aiter.utility'] = MagicMock()
sys.modules['aiter.utility.dtypes'] = MagicMock()
sys.modules['vllm'] = MagicMock()

from atom.audio.chatterbox.engine import RepetitionPenaltyProcessor

def test_repetition_penalty():
    print("Testing RepetitionPenaltyProcessor in-place modification...")
    penalty = 1.2
    processor = RepetitionPenaltyProcessor(penalty=penalty)

    # Create a dummy tensor and an identical copy to check original values
    scores = torch.tensor([[1.0, -1.0, 2.0, 3.0]], dtype=torch.float32)
    original_scores = scores.clone()
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)

    # Apply penalty
    returned_scores = processor(input_ids, scores)

    # Assert it returned the same object
    assert returned_scores is scores, "RepetitionPenaltyProcessor did not return the original tensor"

    # Assert modifications were made in place
    assert not torch.equal(scores, original_scores), "Scores tensor was not modified"
    assert torch.allclose(scores[0, 0], torch.tensor(1.0 / penalty)), f"Expected {1.0 / penalty}, got {scores[0, 0]}"
    assert torch.allclose(scores[0, 1], torch.tensor(-1.0 * penalty)), f"Expected {-1.0 * penalty}, got {scores[0, 1]}"
    print("RepetitionPenaltyProcessor test passed.")

def test_soxr_resample():
    print("Testing soxr resampling...")
    x = np.ones(100, dtype=np.float32)
    y = soxr.resample(x, 16000, 24000)
    assert y.shape == (150,), f"Expected shape (150,), got {y.shape}"
    print("soxr resample test passed.")

if __name__ == "__main__":
    test_repetition_penalty()
    test_soxr_resample()
