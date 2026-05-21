import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch

class RepetitionPenaltyProcessor:
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[0] == 1:
            ids = input_ids[0]
            score = scores[0, ids]
            score.mul_(torch.where(score < 0, self.penalty, 1.0 / self.penalty))
            scores[0, ids] = score
            return scores

        score = torch.gather(scores, 1, input_ids)
        score.mul_(torch.where(score < 0, self.penalty, 1.0 / self.penalty))
        scores.scatter_(1, input_ids, score)
        return scores

def _np_rep_penalty(input_ids, scores, penalty):
    if input_ids.shape[0] == 1:
        ids = input_ids[0]
        s = scores[0, ids]
        mask = s < 0
        s[mask] *= penalty
        s[~mask] /= penalty
        scores[0, ids] = s
        return scores

    score = np.take_along_axis(scores, input_ids, axis=1)
    mask = score < 0
    score[mask] *= penalty
    score[~mask] /= penalty
    np.put_along_axis(scores, input_ids, score, axis=1)
    return scores

# Test torch
processor = RepetitionPenaltyProcessor(1.2)
scores = torch.tensor([[-1.0, 1.0, 2.0]], dtype=torch.float32)
input_ids = torch.tensor([[0, 1]])
res = processor(input_ids, scores)
assert torch.allclose(res, torch.tensor([[-1.2, 0.8333333, 2.0]]))
print("Torch verification passed!")

# Test numpy
scores_np = np.array([[-1.0, 1.0, 2.0]], dtype=np.float32)
input_ids_np = np.array([[0, 1]])
res_np = _np_rep_penalty(input_ids_np, scores_np, 1.2)
assert np.allclose(res_np, np.array([[-1.2, 0.8333333, 2.0]]))
print("Numpy verification passed!")
