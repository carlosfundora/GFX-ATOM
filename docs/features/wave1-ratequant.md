# Wave-1 Feature: RateQuant Bit-Budget Adapter

## Source donor extraction

- Donor: `RateQuant`
- Extracted implementation paths:
  - `ratequant/rate_distortion.py`
  - `ratequant/mixed_precision.py`
  - `ratequant/turboquant_core.py`

## Assimilation target

- `gfxATOM-Rust/python/wave1_donor_adapters.py`
  - `ratequant_optimal_bit_allocation(...)`

## Runtime gating

- Global: `GFXATOM_DONOR_FEATURES=1`
- Feature: `GFXATOM_KV_RATEQUANT=1`
- Policy family: `GFXATOM_KV_POLICY_FAMILY=ratequant`

## Behavior

- Uses reverse-waterfilling-style allocation approximation:
  - higher sensitivity components get higher bit allocations
  - per-component allocations are clamped to `[b_min, b_max]`

## Fallback behavior

- If global or feature flag is off, policy falls back to baseline and emits rejection reason.

