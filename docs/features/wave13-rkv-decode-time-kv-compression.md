# Wave 13 R-KV decode-time KV compression policy

## Source donor

- `.archived/repos/R-KV`

## What was assimilated

- Decode-time redundancy-aware KV compression as a policy surface.
- Observation-token buffering and redundancy-window controls.
- Lambda-weighted importance vs redundancy selection.
- Budgeted top-k retention for reasoning traces.

## Integration path

- Python policy layer: `gfxATOM-Rust/python/kv_policy_arbiter.py`
- Python donor adapter: `gfxATOM-Rust/python/wave1_donor_adapters.py`
- Regression coverage: `gfxATOM-Rust/tests/test_kv_policy_arbiter.py`, `gfxATOM-Rust/tests/test_wave1_donor_adapters.py`

## Enabled behavior

- The policy layer can now describe an R-KV decode-time compression lane.
- Consumers can distinguish reasoning-trace compression from prompt-time quantization.

## Disabled behavior

- `rkv` stays opt-in behind `GFXATOM_DONOR_FEATURES` and `GFXATOM_KV_RKV`.
- No decode kernel or cache manager is changed by this wave.

## Fallback behavior

- If the feature flag is off, requests fall back to baseline policy.
- Invalid parameter values fail fast through helper validation.

## Why this donor matters

- R-KV is one of the few remaining donors with a clearly distinct KV behavior: decoding-time compression for long reasoning traces.
- It complements the existing bit-budget and FP8 KV work without duplicating them.
- The result is a clean policy contract that can be benchmarked later without hard-coding execution details now.
