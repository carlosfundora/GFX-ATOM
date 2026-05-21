# Rust Refactor Benchmark Summary

## Tool Parser Performance

- **Before Command**: `python3 test_tool_parser_perf.py`
- **After Command**: `python3 test_tool_parser_perf.py`
- **Before Timing**: 526.24 ms (for 10 iterations of 5000 tool calls)
- **After Timing**: 170.21 ms (for 10 iterations of 5000 tool calls)
- **Percent Change**: 67.66% improvement (3.1x speedup)
- **Notes on Variance or Limitations**:
  - Input was artificially large (5000 tool calls) to highlight parsing overhead.
  - Rust implementation uses string searching instead of regex execution.
  - Dictionary allocation in PyO3 takes up a notable portion of the remaining time.

