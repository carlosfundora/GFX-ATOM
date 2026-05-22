# Benchmark Summary

- Before Command: `python benchmark_reasoning_isolated.py`
- After Command: `python benchmark_reasoning_rust.py`
- Before Timing: 183.90 ms
- After Timing: 133.19 ms
- Percent Change: -27.57% (27.57% faster)
- Notes: Benchmarked 220,000 small streaming chunks simulating the token-by-token output of an autoregressive reasoning model (containing `<think>` blocks). The new rust native extension for `ReasoningFilter` significantly improves parsing throughput.
