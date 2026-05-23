# Rusty Rust Refactor Report

## Repository Recon
- Found Python tool call streaming parser `ToolCallStreamParser` in `atom/entrypoints/openai/tool_parser.py` which buffers and heavily utilizes `re` logic repeatedly for regex pattern searching upon tokens inside hot loops (`process` buffer).
- Existing Rust bindings are set up in `rust_bindings/` via PyO3 as `atom_rust`.
- The `ReasoningFilter` operates on a somewhat similar principle, being implemented in `rust_bindings/src/reasoning.rs`.
- Decided to rewrite `ToolCallStreamParser` directly into `atom_rust`.

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `ToolCallStreamParser` (`atom/entrypoints/openai/tool_parser.py`) | Python | Higher parsing throughput | Medium | Low | Selected |
| 2 | `parse_tool_calls` (`atom/entrypoints/openai/tool_parser.py`) | Python | Faster full-text parsing | Low | Low | Skipped (already implemented in Rust, fallback exists) |
| 3 | `stable_hash` (`atom/utils/hash.py`) | Python | Faster hashing | Low | Low | Skipped (already uses Rust) |
| 4 | `find_files` (`atom/utils/file_finder.py`) | Python | Faster globbing | Low | Low | Skipped (already uses Rust) |
| 5 | `SentenceSplitter` (`atom/audio/text_splitter.py`) | Python | Faster string split | Low | Low | Skipped (already uses `rs_codec`) |

## Selected Candidate

- Path: `atom/entrypoints/openai/tool_parser.py`
- Current implementation: Pure Python, regex-based substring matching inside a streaming process loop.
- Rust replacement: Pure Rust PyO3 implementation `ToolCallStreamParser` in `rust_bindings/src/tool_parser/mod.rs` avoiding regex string manipulation bottlenecks.
- Reason selected: ToolCall parsing is executed per-token in the hot path of output stream generation (e.g., `serving_chat.py`), so eliminating Regex overhead in Python creates a substantial win.

## Implementation Summary
- Added `ToolCallStreamParser` to `atom_rust` via PyO3.
- Parses state machine in Rust: states 0 (normal), 1 (buffering tool call section), 2 (done).
- Zero-copy compatible PyList emission of `(event_type, data)` mirroring the exact return signature of the original Python code.
- Hooked the Rust implementation inside `atom/entrypoints/openai/tool_parser.py` with fallback safely configured.
- Re-implemented the Rust `process_buffer` to robustly continue upon malformed input tool call blocks (advancing `current_idx`), avoiding permanent parser looping.

## Before Benchmark
See `.jules/verification/rusty/before-benchmark.json`

## After Benchmark
See `.jules/verification/rusty/after-benchmark.json`

## Benchmark Delta
See `.jules/verification/rusty/benchmark-summary.md`

## Tests Run
- `pytest tests/entrypoints/test_tool_parser.py`

## Files Changed
- `atom/entrypoints/openai/tool_parser.py`
- `rust_bindings/src/tool_parser/mod.rs`
- `rust_bindings/src/lib.rs`

## Compatibility Notes
- Pure-python version retained fully functioning as a fallback inside a `try/except ImportError` block.

## Remaining Follow-Ups
- None. PR is ready for merge.
