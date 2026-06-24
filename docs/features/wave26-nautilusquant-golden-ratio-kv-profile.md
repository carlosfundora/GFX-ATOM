# Wave-26 Nautilus golden-ratio KV profile

## Source donor extraction

- Donor: `NautilusQuant`
- Extracted implementation idea:
  - deterministic orthogonal KV quantization with golden-ratio geometry

## Assimilation target

- `gfxATOM-Rust/python/wave1_donor_adapters.py`
  - `nautilus_geometric_profile(...)`
- `gfxATOM-Rust/python/kv_policy_arbiter.py`
  - `Wave1PolicyProfile.nautilus`

## Runtime gating

- Global: `GFXATOM_DONOR_FEATURES=1`
- Family flag: `GFXATOM_KV_NAUTILUS=1`

## Behavior

- Uses a golden-ratio ladder to model deterministic geometric weighting.
- Hooks into the existing outlier gate already used by the adaptive recommendation lane.
- Exposes a compact profile object without changing backend execution paths.

## Fallback behavior

- If the feature flag is off, the arbiter falls back to baseline selection.
- If `nautilus` is not selected, the profile field stays absent.

## Why this donor matters

- Nautilus brings a deterministic geometry flavor to the KV policy stack.
- It complements the existing ratequant, delta-k, wobble, and QAQ policy lanes.
- The result is a reproducible policy surface that can be benchmarked later without extra runtime complexity.
