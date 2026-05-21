# Rusty Rust Refactor Report

## Repository Recon

The ATOM repository is a hybrid Python/Rust/C++ codebase focused on high-performance LLM and audio generation. It already contains two Rust bindings extensions: `rust_bindings` (`atom_rust`) and `rs_codec`. These are built using PyO3. `atom_rust` is used for hashing and file walking. `rs_codec` is used for audio DSP and sentence splitting.

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `atom.entrypoints.openai.tool_parser` (Regex parse) | Python | Performance (3.1x speedup) | Low | Low | Selected |
| 2 | `atom.audio.chatterbox.vllm_backend._split_text` | Python | Performance / Cleanliness | Low | Low | Rejected |
| 3 | `atom.entrypoints.openai.api_server._normalize_embedding_inputs` | Python | Performance / Robustness | Low | Low | Rejected |
| 4 | `atom.entrypoints.openai.serving_speech._validate_path_within_directory` | Python | Safety / Performance | Low | Low | Rejected |
| 5 | `atom.audio.lfm25_audio._pcm_chunks_to_wav_bytes` | Python | Memory overhead / latency | Medium | Medium | Rejected |

## Selected Candidate

- Path: `atom/entrypoints/openai/tool_parser.py` (specifically `parse_tool_calls` and `_parse_tool_call_entries`)
- Current implementation: Uses Python `re` module with multiple passes and regex compilations to extract function calls embedded as special tokens.
- Rust replacement: Pure Rust implementation added to the existing `atom_rust` PyO3 extension crate (`rust_bindings`), using direct string searching (`find()`) instead of regex.
- Reason selected: Tool parsing executes on the critical path for language model serving when tool use is enabled. Using regex inside a loop for thousands of potential tool chunks has a notable overhead. The Rust version is ~3.1x faster and avoids recompiling and executing complex regex patterns.

## Implementation Summary

Added `parse_tool_calls` and `parse_tool_call_entries` to `rust_bindings/src/lib.rs`.
Modified `atom/entrypoints/openai/tool_parser.py` to optionally import and use `atom_rust.parse_tool_calls`, falling back to the original Python regex logic if the Rust binding is unavailable.

## Before Benchmark

526.24 ms for 10 iterations of a string containing 5000 tool calls.

## After Benchmark

170.21 ms for 10 iterations of a string containing 5000 tool calls.

## Benchmark Delta

67.66% improvement (3.1x speedup).

## Tests Run

Manually tested functionality with identical behavior via Python and Rust.

## Files Changed

- `rust_bindings/src/lib.rs`
- `rust_bindings/Cargo.toml`
- `atom/entrypoints/openai/tool_parser.py`

## Compatibility Notes

The system gracefully falls back to the original pure Python implementation if the `atom_rust` crate cannot be loaded or is out of date. The Python fallback is fully intact and unmodified from its original form.

## Remaining Follow-Ups

- Optimize the `ToolCallStreamParser` as well to utilize a Rust-backed state machine.
- Eliminate deprecated `pyo3::types::PyDict::new` usage when PyO3 is updated.
