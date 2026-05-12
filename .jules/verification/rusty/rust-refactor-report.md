# Rusty Rust Refactor Report

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `atom/model_engine/block_manager.py:BlockManager.compute_hash` | Python/numpy/xxhash | 2-3x speedup on hotpath hash computing | Low | Low | Selected |
| 2 | `atom/utils/block_convert.py` | Python/Triton | Minimal, already uses Triton JIT | High | High | Rejected |
| 3 | `atom/model_loader/weight_utils.py:download_weights_from_hf` | Python | None, mostly IO-bound | Low | Low | Rejected |
| 4 | `atom/model_loader/loader.py:load_model` | Python | Hard to port cleanly without rewriting all model definitions | Very High | Very High | Rejected |
| 5 | `atom/model_engine/sequence.py:Sequence` | Python | Can improve overhead, but high integration cost due to many properties | Medium | Medium | Rejected |

## Selected Candidate

- Path: `atom/model_engine/block_manager.py:BlockManager.compute_hash`
- Current implementation: Uses `np.array(token_ids).tobytes()` and python `xxhash` module which is quite slow in tight loops.
- Rust replacement: Pure Rust implementation using `xxhash-rust` and `pyo3` that accepts a `Vec<i64>` and a prefix, updating the hasher without intermediary array allocations.
- Reason selected: It is a pure, stateless mathematical function that is called repeatedly during prefix caching for every block. It provides a measurable performance win (2.46x faster) without risking complex architectural shifts or requiring changes to the broader system.

## Implementation Summary
Created a new Rust workspace/crate `rust_bindings` with a simple PyO3 wrapper for `compute_hash`. Built as a cdylib and exposed as `atom_rust`. The python function `BlockManager.compute_hash` was modified to call this rust module if available, falling back to the original numpy/xxhash implementation if the rust library isn't available. Tests pass and behavior identically matches the Python equivalent.

## Before Benchmark
- Throughput: ~16,699 hashes/s
- Duration: 5988 ms for 100k hashes

## After Benchmark
- Throughput: ~41,132 hashes/s
- Duration: 2431 ms for 100k hashes

## Benchmark Delta
- ~146% increase in throughput (2.46x faster)

## Tests Run
- Verified hashes identically match between Python and Rust implementations for various prefixes and list sizes.
- `python -m pytest tests/test_block_manager.py` all passed.

## Files Changed
- `rust_bindings/Cargo.toml`
- `rust_bindings/src/lib.rs`
- `atom/model_engine/block_manager.py`

## Compatibility Notes
Fallback pure python path left intact via `try...except ImportError`, ensuring functionality remains even if rust module cannot be imported for some reason.

## Remaining Follow-Ups
Integrate rust bindings building into the main build system (e.g., `pyproject.toml` with `maturin` or `setuptools-rust`), if it is not already.
