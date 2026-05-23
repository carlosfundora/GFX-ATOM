# Benchmark Summary

- Before command: `python test_np_rep_penalty_benchmark.py` (which uses pure NumPy `take_along_axis` and `put_along_axis`)
- After command: `python test_np_rep_penalty_benchmark.py` (which uses the custom `rs_codec.rep_penalty_kernel` Zero-Copy SIMD-capable backend)
- Before timing: ~464 ms for 10000 iterations of batched repetition penalty on a vocab of 32000 and seq_len of 512.
- After timing: ~78 ms for 10000 iterations of the same workload.
- Percent change: ~83% reduction in execution time (nearly 6x faster logic).
- Notes: The python hot loop in Chatterbox Engine relies heavily on generating tokens individually using a repetition penalty across the entire logits tensor. Moving the index iteration, deduplication, and conditional logic (where score < 0) out of NumPy operations into Rust with PyO3 bindings allows this operation to be in-place zero-copy and removes the heavy array allocation overhead inherent to numpy `where` and `take_along_axis`.
