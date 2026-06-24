# Wave-24 wobble-quant-cache variance-weighted KV allocation

## Source donor extraction

- Donor: `wobble-quant-cache`
- Extracted implementation idea:
  - variance-aware per-dimension KV bit allocation

## Assimilation target

- `gfxATOM-Rust/python/wave1_donor_adapters.py`
  - `wobble_variance_bit_allocation(...)`
- `gfxATOM-Rust/python/kv_policy_arbiter.py`
  - `Wave1PolicyProfile.wobble`

## Runtime gating

- Global: `GFXATOM_DONOR_FEATURES=1`
- Family flag: `GFXATOM_KV_WOBBLE=1`

## Behavior

- Allocates more bits to higher-variance lanes.
- Keeps a compact, deterministic profile object for policy selection.
- Reuses the existing wave-1 donor arbiter instead of adding a new backend.

## Fallback behavior

- If the feature flag is off, the arbiter falls back to baseline selection.
- If `wobble` is not selected, the profile field stays absent.

## Why this donor matters

- `wobble-quant-cache` is a good fit for dimension-aware mixed-precision policies.
- It fills the gap between rate-distortion allocation and other cache compression heuristics.
- The result is a small but practical policy lane that can be benchmarked later without runtime churn.
