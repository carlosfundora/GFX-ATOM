import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from atom.audio.chatterbox.engine import RepetitionPenaltyProcessor, ChatterboxEngine
import torch
import numpy as np

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
res_np = ChatterboxEngine._np_rep_penalty(input_ids_np, scores_np, 1.2)
assert np.allclose(res_np, np.array([[-1.2, 0.8333333, 2.0]]))
print("Numpy verification passed!")
