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

if __name__ == "__main__":
    benchmark_text_splitting()
