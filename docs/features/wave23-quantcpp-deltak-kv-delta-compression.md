# Wave-23 quant.cpp delta-k KV delta compression profile

## Source donor extraction

- Donor: `quant.cpp`
- Extracted implementation paths:
  - `include/turboquant/tq_engine.h`
  - `quant.h`
  - `src/engine/tq_generate.c`
  - `src/engine/tq_transformer.c`
  - `src/server/tq_server.c`

## Assimilation target

- `gfxATOM-Rust/python/wave1_donor_adapters.py`
  - `delta_kv_profile(...)`
- `gfxATOM-Rust/python/kv_policy_arbiter.py`
  - `Wave1PolicyProfile.delta_kv`

## Runtime gating

- Global: `GFXATOM_DONOR_FEATURES=1`
- Family flag: `GFXATOM_KV_DELTAK=1`

## Behavior

- Exposes delta-K as a small explicit profile lane.
- Captures the key-delta compression mode as a first-class policy metadata object.
- Preserves the existing fail-closed policy arbitration path.

## Fallback behavior

- If the flag is off, the policy arbiter falls back to baseline selection.
- The delta-K profile stays absent unless the `deltak` family is selected.

## Why this donor matters

- `quant.cpp`’s delta-K mode is a compact KV-compression idea with a small surface area.
- It fits the existing wave-1 policy arbiter without requiring a new runtime backend.
- The result is a clean, benchmarkable policy lane that can be expanded later if the donor proves useful on gfx1030.
