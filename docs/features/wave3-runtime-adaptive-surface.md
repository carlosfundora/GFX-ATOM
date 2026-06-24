# Wave-3 Feature: Runtime Adaptive Recommendation Surface

## Purpose

Expose adaptive policy recommendation in runtime profile payload so dashboard/control-plane consumers can inspect recommendation context alongside backend/cache capability state.

## Assimilation targets

- Python:
  - `gfxATOM-Rust/python/engine_runtime_profile.py`
  - new field: `adaptive_recommendation`
  - new helper: `with_adaptive_recommendation(...)`
- Rust parity:
  - `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`
  - new field: `adaptive_recommendation: Option<String>`
  - new helper: `with_adaptive_recommendation(...)`

## Behavior

- Field is optional and defaults to `None`.
- Intended payload shape:
  - `family`
  - `score`
  - `reason`
- Surface is read-only metadata for observability and operator review.

## Safety boundary

- Runtime profile does not execute or force policy changes from this field.
- Selection remains governed by policy arbiter/feature-gate path.

