# Wave-28 autoquant observer snapshot parity

## Source donor extraction

- Donor: `sglang-1-bit-turbo`
- Extracted implementation idea:
  - JSON-safe observer snapshot contract for autoquant calibration telemetry

## Assimilation target

- `gfxATOM-Rust/crates/rs_autoquant_policy/src/lib.rs`
- `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs`

## Runtime gating

- This is a validation/parity lane, not a runtime switch.
- It keeps the observer data model aligned with the SGLang donor surface.

## Behavior

- `AutoQuantObserverSnapshot` round-trips cleanly through Rust serialization.
- The validation harness now checks the observer snapshot shape alongside the policy shelf round-trip.

## Fallback behavior

- No runtime path changes.
- Existing autoquant policy behavior remains unchanged if the snapshot is ignored.

## Why this donor matters

- The observer snapshot is the missing half of the autoquant donor contract.
- Keeping policy and observer parity in Rust makes later Rust-first calibration work easier to validate.
- This gives gfxATOM a stable, serializable telemetry shape without dragging in the Python observer hot path.
