# Rusty Rust Refactor Report

## Repository Recon
Found Python-based string parsers in the hot path of autoregressive generation loops: `ToolCallStreamParser` and `ReasoningFilter` inside `atom/entrypoints/openai/`.
The memory implicitly requested a native Rust implementation via PyO3 inside `atom_rust` to avoid inefficient substring allocations and regex overhead. However, the `atom_rust` crate did not exist.

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `ToolCallStreamParser` in `atom/entrypoints/openai/tool_parser.py` | Python | Avoid regex and string slicing in hot loops | Medium | Low | Select |
| 2 | `ReasoningFilter` in `atom/entrypoints/openai/reasoning.py` | Python | Avoid regex and string slicing in hot loops | Medium | Low | Select |

## Selected Candidate

- Path: `atom/entrypoints/openai/tool_parser.py` and `atom/entrypoints/openai/reasoning.py`
- Current implementation: Pure Python regex and string manipulation.
- Rust replacement: `atom_rust` PyO3 module.
- Reason selected: Highly sensitive token streaming paths that benefit greatly from native string parsing, up to ~2x speedup.

## Implementation Summary
Created a new PyO3 module `atom_rust` inside `crates/rs_atom_entrypoints_parsing` using `maturin build --release`. Implemented `ToolCallStreamParser` and `ReasoningFilter` inside Rust using pure rust string parsing logic. Updated the python modules to import the native module, gracefully falling back to the python version if importing fails.

## Before Benchmark
Saved to `.jules/verification/rusty/before-benchmark.json`

## After Benchmark
Saved to `.jules/verification/rusty/after-benchmark.json`

## Benchmark Delta
- ToolCallStreamParser: 2.01x speedup
- ReasoningFilter: 1.75x speedup

## Tests Run
- `pytest tests/entrypoints/test_reasoning.py` (Passed)
- `pytest tests/entrypoints/test_tool_parser.py` (Passed)

## Files Changed
- `crates/rs_atom_entrypoints_parsing/Cargo.toml`
- `crates/rs_atom_entrypoints_parsing/src/lib.rs`
- `atom/entrypoints/openai/tool_parser.py`
- `atom/entrypoints/openai/reasoning.py`

## Compatibility Notes
Fallback to pure Python implementations are explicitly preserved if the `atom_rust` wheel is missing or fails to import.

## Remaining Follow-Ups
None.
