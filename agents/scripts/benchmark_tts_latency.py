import time
import sys
import numpy as np

try:
    import rs_codec
    has_rs_codec = True
except ImportError:
    has_rs_codec = False

def benchmark_text_splitting():
    print(f"rs_codec available: {has_rs_codec}")

    # Pure Python baseline (mocked to represent pre-rust code that was removed)
    text = "Hello there. How are you doing today? I am doing well! This is a test of the text splitting algorithm." * 100

    t0 = time.perf_counter()
    chunks = []
    buffer = ""
    for char in text:
        buffer += char
        if char in {'.', '!', '?'}:
            chunks.append(buffer)
            buffer = ""
    t1 = time.perf_counter()
    py_time = (t1 - t0) * 1000
    print(f"Python text split baseline: {py_time:.2f} ms")

    if has_rs_codec:
        t2 = time.perf_counter()
        splitter = rs_codec.SentenceSplitter(min_sentence_length=2)
        rust_chunks = splitter.add_text(text)
        t3 = time.perf_counter()
        rs_time = (t3 - t2) * 1000
        print(f"Rust text split (rs_codec): {rs_time:.2f} ms")
        print(f"Speedup: {py_time / rs_time:.2f}x")

import torch

def benchmark_rep_penalty_numpy():
    vocab_size = 32000
    batch_size = 1
    seq_len = 500

    def new_in_place(input_ids, scores, penalty):
        if input_ids.shape[0] == 1:
            ids = input_ids[0]
            s = scores[0, ids]
            mask = s < 0
            s[mask] *= penalty
            s[~mask] /= penalty
            scores[0, ids] = s
            return scores
        return scores

    t0 = time.perf_counter()
    for _ in range(1000):
        scores = np.random.randn(batch_size, vocab_size)
        input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        new_in_place(input_ids, scores, 1.2)
    py_time_new = (time.perf_counter() - t0) * 1000
    print(f"NumPy rep penalty: {py_time_new:.2f} ms")

def benchmark_rep_penalty_torch():
    vocab_size = 32000
    batch_size = 1
    seq_len = 500

    def new_in_place(input_ids, scores, penalty):
        if input_ids.shape[0] == 1:
            ids = input_ids[0]
            s = scores[0, ids]
            s.mul_(torch.where(s < 0, penalty, 1.0 / penalty))
            scores[0, ids] = s
            return scores
        return scores

    t0 = time.perf_counter()
    for _ in range(1000):
        scores = torch.randn(batch_size, vocab_size, device="cpu")
        input_ids = torch.randint(0, vocab_size, size=(batch_size, seq_len), device="cpu")
        new_in_place(input_ids, scores, 1.2)
    py_time_new = (time.perf_counter() - t0) * 1000
    print(f"Torch rep penalty: {py_time_new:.2f} ms")

def main():
    benchmark_text_splitting()
    benchmark_rep_penalty_numpy()
    benchmark_rep_penalty_torch()

if __name__ == '__main__':
    main()
