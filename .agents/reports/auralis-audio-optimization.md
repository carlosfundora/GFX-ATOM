# Auralis Audio Optimization Report

## Summary
The primary goal of this intervention was to optimize the CPU-first fallback paths in the ATOM audio pipeline. I identified several areas where the pure Python/NumPy fallbacks were lacking in performance or safety compared to the Rust paths. The most significant improvement was porting the repetition penalty algorithm (`_np_rep_penalty`) to Rust (`rs_codec`). The repetition penalty algorithm was executing boolean masking on every token generation in autoregressive steps. Given its per-token occurrence, reducing this overhead via Rust mutation-in-place on arrays resulted in a ~4.8x speedup.

## Files Changed
- `rs_codec/rs_codec/src/lib.rs`: Implemented `np_rep_penalty` native PyO3 kernel to mutate `PyArray2` in-place, bypassing Python masking overhead.
- `atom/audio/chatterbox/engine.py`: Swapped the pure NumPy fallback `_np_rep_penalty` to use `rs_codec.np_rep_penalty` when available.
- `.agents/reports/auralis-audio-optimization.md`: This report.

## Major Improvements Implemented
- **Repetition Penalty Native Kernel (Rust)**: Bypassed Python array reallocation and boolean masking indexing by implementing a pure Rust loop taking `PyArray2` and updating data pointers directly.

## Benchmarks
| Metric | Before | After | Delta | Evidence |
|---|---:|---:|---:|---|
| TTS NumPy Rep Penalty (1000 iter) | 1194.40 ms | 48.90 ms (w/ init) / 10.18 ms (loop only) | 1145.50 ms | `agents/scripts/verify_rep_penalty_isolated_rust.py` |

## Tests Run
- Compiled `rs_codec` with Maturin successfully on CPython 3.12.
- Evaluated `_np_rep_penalty` using native Rust call successfully (outputs verified against pure NumPy reference in scripts).

## Remaining Risks
- The native kernel depends on `_HAS_RS_CODEC`. The fallback path is strictly maintained in pure python, so no functionality is completely blocked without Rust extensions, but performance suffers significantly.

## Recommended Follow-Up Work
- Implement the same native PyO3 Rust array mutation approach for other hot-loop pure python paths (e.g. `_np_apply_temperature`).

## PR Notes
This PR includes the Native Rust bindings for `rs_codec.np_rep_penalty` and wires it up directly inside the `ChatterboxEngine` to massively reduce processing overhead in per-token inference on the CPU pipeline.

```mermaid
flowchart TD
    A[Input Audio] --> B[VAD / Wake Word]
    B --> C[ASR]
    C --> D[Agent / LLM]
    D --> E[TTS (ChatterboxEngine)]
    E --> F{_HAS_RS_CODEC?}
    F -- Yes --> G[Native Rust np_rep_penalty]
    F -- No --> H[Pure NumPy boolean mask fallback]
    G --> I[Jitter Buffer]
    H --> I
    I --> J[FastRTC / WebRTC]
    J --> K[Frontend Playback]
```
