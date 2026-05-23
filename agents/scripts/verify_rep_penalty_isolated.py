import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import time

def np_where_penalty(input_ids, scores, penalty):
    if input_ids.shape[0] == 1:
        ids = input_ids[0]
        s = scores[0, ids]
        s = np.where(s < 0, s * penalty, s / penalty)
        scores[0, ids] = s
        return scores
    score = np.take_along_axis(scores, input_ids, axis=1)
    score = np.where(score < 0, score * penalty, score / penalty)
    np.put_along_axis(scores, input_ids, score, axis=1)
    return scores

def np_mask_penalty(input_ids, scores, penalty):
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

vocab_size = 32000
batch_size = 1
seq_len = 500
penalty = 1.2

for name, func in [("np_where", np_where_penalty), ("np_mask", np_mask_penalty)]:
    t0 = time.perf_counter()
    for _ in range(1000):
        scores = np.random.randn(batch_size, vocab_size).astype(np.float32)
        input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        func(input_ids, scores, penalty)
    print(f"{name}: {(time.perf_counter() - t0)*1000:.2f} ms")
