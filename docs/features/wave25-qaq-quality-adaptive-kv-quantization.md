# Wave-25 QAQ quality-adaptive KV quantization

## Source donor extraction

- Donor: `QAQ-KVCacheQuantization`
- Extracted implementation idea:
  - quality-adaptive KV quantization with minimal-loss target selection

## Assimilation target

- `gfxATOM-Rust/python/wave1_donor_adapters.py`
  - `qaq_quality_profile(...)`
- `gfxATOM-Rust/python/kv_policy_arbiter.py`
  - `Wave1PolicyProfile.qaq`

## Runtime gating

- Global: `GFXATOM_DONOR_FEATURES=1`
- Family flag: `GFXATOM_KV_QAQ=1`

## Behavior

- Interprets the wave-1 signal vector as a quality proxy for QAQ decisions.
- Compares the observed quality against a target and exposes a compact profile object.
- Keeps the policy lane fail-closed and benchmarkable without changing runtime execution.

## Fallback behavior

- If the feature flag is off, the arbiter falls back to baseline selection.
- If `qaq` is not selected, the profile field stays absent.

## Why this donor matters

- QAQ covers the quality-adaptive side of KV quantization without requiring a new backend.
- It complements the existing ratequant, wobble, delta-k, and R-KV policy lanes.
- The resulting contract is small enough to benchmark and extend later without extra plumbing.
