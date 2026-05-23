import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import time

def np_loop_penalty(input_ids, scores, penalty):
    if input_ids.shape[0] == 1:
        ids = input_ids[0]
        # modify specific elements
        for i in ids:
            val = scores[0, i]
            if val < 0:
                scores[0, i] = val * penalty
            else:
                scores[0, i] = val / penalty
        return scores
    return scores

vocab_size = 32000
batch_size = 1
seq_len = 500
penalty = 1.2

for name, func in [("np_loop", np_loop_penalty)]:
    t0 = time.perf_counter()
    for _ in range(1000):
        scores = np.random.randn(batch_size, vocab_size).astype(np.float32)
        input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        func(input_ids, scores, penalty)
    print(f"{name}: {(time.perf_counter() - t0)*1000:.2f} ms")
