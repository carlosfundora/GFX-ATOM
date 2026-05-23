import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import time
import rs_codec

vocab_size = 32000
batch_size = 1
seq_len = 500
penalty = 1.2

# We need to subtract random array creation time
scores_list = [np.random.randn(batch_size, vocab_size).astype(np.float32) for _ in range(1000)]
input_ids_list = [np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64) for _ in range(1000)]

t0 = time.perf_counter()
for i in range(1000):
    rs_codec.np_rep_penalty(scores_list[i], input_ids_list[i], penalty)
print(f"rust_rep_penalty loop only: {(time.perf_counter() - t0)*1000:.2f} ms")

t0 = time.perf_counter()
for i in range(1000):
    scores = scores_list[i]
    input_ids = input_ids_list[i]
    if input_ids.shape[0] == 1:
        ids = input_ids[0]
        s = scores[0, ids]
        mask = s < 0
        s[mask] *= penalty
        s[~mask] /= penalty
        scores[0, ids] = s
    else:
        score = np.take_along_axis(scores, input_ids, axis=1)
        mask = score < 0
        score[mask] *= penalty
        score[~mask] /= penalty
        np.put_along_axis(scores, input_ids, score, axis=1)
print(f"numpy_rep_penalty loop only: {(time.perf_counter() - t0)*1000:.2f} ms")
