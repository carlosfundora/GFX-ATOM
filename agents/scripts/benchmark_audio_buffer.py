import time
import numpy as np

def benchmark_buffer_alloc():
    # Simulate assembling continuous audio data from multiple chunks
    audio_chunks = [np.random.randn(24000).astype(np.float32) for _ in range(50)]

    t0 = time.perf_counter()
    for _ in range(100):
        result = np.array([], dtype=np.float32)
        for chunk in audio_chunks:
            result = np.concatenate((result, chunk))
    t1 = time.perf_counter()
    print(f"Iterative concatenation (np.concatenate): {(t1 - t0) * 1000:.2f} ms")

    t0 = time.perf_counter()
    for _ in range(100):
        total_len = sum(len(c) for c in audio_chunks)
        result = np.empty(total_len, dtype=np.float32)
        offset = 0
        for chunk in audio_chunks:
            result[offset:offset+len(chunk)] = chunk
            offset += len(chunk)
    t1 = time.perf_counter()
    print(f"Pre-allocation (np.empty + slices): {(t1 - t0) * 1000:.2f} ms")

if __name__ == "__main__":
    benchmark_buffer_alloc()
