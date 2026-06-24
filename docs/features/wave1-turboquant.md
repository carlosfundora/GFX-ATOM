# Wave-1 Feature: TurboQuant Profile Adapter

## Source donor extraction

- Donor: `turbo-quant`
- Extracted implementation paths:
  - `src/turbo.rs`
  - `src/qjl.rs`
  - `src/kv.rs`
  - `src/polar.rs`

## Assimilation target

- `gfxATOM-Rust/python/wave1_donor_adapters.py`
  - `turboquant_profile(...)`

## Runtime gating

- Global: `GFXATOM_DONOR_FEATURES=1`
- Common use with policy families:
  - `ratequant`
  - `deltak`

## Behavior

- Encodes donor split model as profile metadata:
  - Polar stage = `b-1` bits/value
  - QJL residual stage = `1` bit/value
- Uses a default projection heuristic (`dim/4`) when not explicitly provided.

## Fallback behavior

- Invalid vector shape or bit-width raises explicit validation error.
- Unsupported policy family path keeps baseline behavior.

