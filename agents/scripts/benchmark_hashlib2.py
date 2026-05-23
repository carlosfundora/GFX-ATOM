import sys
import hashlib
import time

try:
    import atom_rust
    has_rust = True
except ImportError:
    has_rust = False

def python_compute_hash_str(factors):
    hash_str = hashlib.md5(
        str(factors).encode(), usedforsecurity=False
    ).hexdigest()[:10]
    return hash_str

def run_bench():
    factors = {"a": 1, "b": "test", "c": [1, 2, 3], "d": {"nested": "value"}}

    t0 = time.perf_counter()
    for _ in range(100000):
        python_compute_hash_str(factors)
    t1 = time.perf_counter()
    py_time = (t1 - t0) * 1000

    # We will implement compute_hash_str in rust
    t0 = time.perf_counter()
    for _ in range(100000):
        atom_rust.compute_string_hash(str(factors))[:10]
    t1 = time.perf_counter()
    rs_time = (t1 - t0) * 1000

    print(f"Python duration: {py_time:.2f} ms")
    print(f"Rust duration: {rs_time:.2f} ms")
    print(f"Speedup: {py_time / rs_time:.2f}x")

if __name__ == "__main__":
    run_bench()
