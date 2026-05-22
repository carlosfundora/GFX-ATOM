# Rusty Rust Refactor Report

## Repository Recon

The ATOM repository is a vLLM-like implementation focusing on integration and optimization based on AITER. It exposes an OpenAI-compatible API Server (`atom/entrypoints/openai_server.py`). The repository relies extensively on a native Rust extension `atom_rust` located under `rust_bindings/` to accelerate hot path tasks such as token ID hashing, string hashing, fast file scanning, and streaming tool call extraction (`parse_tool_calls` in `atom/entrypoints/openai/tool_parser.py`).

In `atom/entrypoints/openai/reasoning.py`, a `ReasoningFilter` class is used similarly to `ToolCallStreamParser` to separate `<think>` blocks from the normal chat completions output text. This logic is invoked directly within the asynchronous generation tight loop (`create_chat_chunk` iteration within `serving_chat.py`). Since string allocations and manipulations in Python can become a bottleneck when processing tokens one by one, migrating this streaming parser to Rust provides a valuable latency reduction.

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `ReasoningFilter` (`atom/entrypoints/openai/reasoning.py`) | Python | Lower latency in token streaming loop | Medium | Low | Selected |
| 2 | `ToolCallStreamParser` (`atom/entrypoints/openai/tool_parser.py`) | Python | Lower latency in token streaming loop | High | Medium | Rejected (already has `parse_tool_calls` in Rust, streaming parser state machine complex due to JSON accumulation) |
| 3 | `stable_hash` (`atom/utils/hash.py`) | Python fallback | High performance hashing | Low | Low | Rejected (already uses Rust under the hood via `atom_rust.compute_bytes_hash`) |
| 4 | `mean_pool_embeddings` (`atom/retrieval/colbert.py`) | Python (PyTorch) | Lower memory/latency | Medium | Medium | Rejected (memory notes say "Avoid rewriting PyTorch-based tensor operations in pure Rust.") |
| 5 | `maxsim_score` (`atom/retrieval/colbert.py`) | Python (PyTorch) | Lower memory/latency | Medium | Medium | Rejected (memory notes say "Avoid rewriting PyTorch-based tensor operations in pure Rust.") |

## Selected Candidate

- Path: `atom/entrypoints/openai/reasoning.py`
- Current implementation: `ReasoningFilter` (pure Python state machine using `str.split()` and slicing).
- Rust replacement: Add `ReasoningFilter` pyclass to `atom_rust` crate.
- Reason selected: Directly executed in the token generation streaming hot loop (`serving_chat.py`). Pure string matching without requiring external dependencies or async boundaries.

## Implementation Summary

Added `ReasoningFilter` as a `#[pyclass]` in the existing `rust_bindings` crate. It stores a state (`u8`) and a buffer (`String`). The `process` method modifies the internal state and string buffer and yields a `PyList` of `(field_name, text)` tuples. The `flush` method emits any remaining buffered text. The Python `ReasoningFilter` has been updated to dynamically import `atom_rust` and delegate execution to the native class if available, maintaining strict backward compatibility if the compiled extension isn't present.

## Before Benchmark
- `python benchmark_reasoning_isolated.py`
- 183.90 ms (220k iterations)

## After Benchmark
- `python benchmark_reasoning_rust.py`
- 133.19 ms (220k iterations)

## Benchmark Delta
- -27.57% (27.57% faster execution of the text parsing loop)

## Tests Run
- `pytest tests/entrypoints/test_reasoning.py`
- Pass

## Files Changed
- `rust_bindings/src/lib.rs`
- `rust_bindings/src/reasoning.rs`
- `atom/entrypoints/openai/reasoning.py`

## Compatibility Notes
- The Python implementation has been preserved as a transparent fallback in `ReasoningFilter` when `atom_rust` is not installed or when `atom_rust.ReasoningFilter` is absent.

## Remaining Follow-Ups
- None.
