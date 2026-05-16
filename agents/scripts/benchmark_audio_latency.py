import time
import sys
import numpy as np

try:
    import rs_codec
    has_rs_codec = True
except ImportError:
    has_rs_codec = False

def benchmark_pcm_conversion():
    print(f"rs_codec available: {has_rs_codec}")

    # Create 1 minute of fake audio at 24kHz
    sample_rate = 24000
    duration = 60
    audio = np.random.uniform(-1.0, 1.0, sample_rate * duration).astype(np.float32)

    # Pure Python
    t0 = time.perf_counter()
    pcm_data = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    pcm_bytes_py = pcm_data.tobytes()
    t1 = time.perf_counter()
    py_time = (t1 - t0) * 1000
    print(f"Python PCM conversion: {py_time:.2f} ms")

    if has_rs_codec:
        t2 = time.perf_counter()
        pcm_bytes_rs = rs_codec.audio_to_pcm_bytes(audio)
        t3 = time.perf_counter()
        rs_time = (t3 - t2) * 1000
        print(f"Rust PCM conversion: {rs_time:.2f} ms")
        print(f"Speedup: {py_time / rs_time:.2f}x")

def benchmark_agc():
    print(f"rs_codec available: {has_rs_codec}")

    sample_rate = 24000
    duration = 60
    audio = np.random.uniform(-1.0, 1.0, sample_rate * duration).astype(np.float32)

    if has_rs_codec:
        t2 = time.perf_counter()
        rs_codec.agc_kernel(audio, 0.125, 0.01, 0.1, 10.0, 2400, 1.0)
        t3 = time.perf_counter()
        rs_time = (t3 - t2) * 1000
        print(f"Rust AGC conversion: {rs_time:.2f} ms")

if __name__ == "__main__":
    benchmark_pcm_conversion()
    benchmark_agc()
