# Prefix Cache Hash Benchmark Summary

* **Before Command:** `python benchmark_block_manager_hash.py`
* **After Command:** `python benchmark_rust_hash.py`
* **Before Timing:** 5988 ms (100,000 iterations, ~16,699 hashes/s)
* **After Timing:** 2431 ms (100,000 iterations, ~41,132 hashes/s)
* **Percent Change:** Throughput increased by 146.3% (2.46x faster)

**Notes on Variance/Limitations:**
This benchmark measures just the raw `compute_hash` function overhead which gets called heavily in the python hot path when prefix caching is enabled (every time a block is allocated or checked). The python version calls `np.array(token_ids).tobytes()` on every call, which involves list traversal and allocation. The Rust version iterates the integers directly and hashes them efficiently using `xxhash-rust`.
