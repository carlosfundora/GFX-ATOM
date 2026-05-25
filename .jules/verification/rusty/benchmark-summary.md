# Benchmark Summary

## Command
`python3 benchmarks/bench_tool_parser.py`

## ToolCallStreamParser
- Before timing (Python): 1.14663s
- After timing (Rust): 0.57839s
- Percent change: -49.5%
- Speedup: 1.98x

## ReasoningFilter
- Before timing (Python): 0.60066s
- After timing (Rust): 0.51486s
- Percent change: -14.3%
- Speedup: 1.17x

## Notes
The Rust implementation provides significant speedup for both `ToolCallStreamParser` and `ReasoningFilter`, avoiding regex execution overhead and inefficient substring allocations.
