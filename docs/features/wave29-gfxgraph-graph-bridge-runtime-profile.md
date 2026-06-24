# Wave-29 gfxGRAPH graph-bridge runtime profile

## Source donor extraction

- Donor: `gfxGRAPH`
- Extracted implementation idea:
  - HIP Graph parity features for conditional nodes, validation, nested capture, and shape bucketing

## Assimilation target

- `gfxATOM-Rust/python/engine_runtime_profile.py`
- `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`
- `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py`

## Runtime gating

- This is a capability-profile lane, not an execution-path toggle.
- It advertises graph-bridge support without changing the capture/replay engine.

## Behavior

- Adds explicit bridge flags for graph shape bucketing, validation mode, conditional nodes, and nested capture.
- Keeps the Python and Rust runtime-profile contracts in lockstep.
- Gives downstream routing code a clean way to reason about gfxGRAPH-capable deployments.

## Fallback behavior

- All new flags default to `false`.
- Existing runtime consumers remain unaffected unless they inspect the new fields.

## Why this donor matters

- gfxGRAPH is the dedicated graph-capture donor for RDNA2/ROCm parity gaps.
- Surfacing its capabilities in the runtime profile makes later routing and benchmark work easier to gate cleanly.
- This is the smallest useful seam for carrying gfxGRAPH features into the engine contract without pulling in the full bridge implementation yet.
