# Wave 9 llama-swap routing capability profile

## Source donor

- `.archived/repos/llama-swap`

## What was assimilated

- Hot-swap model lifecycle as a capability surface.
- Model aliases and direct upstream model naming as routing controls.
- Grouped residency, TTL unload, request filters, and config reload as control-plane capabilities.

## Integration path

- Python contract: `gfxATOM-Rust/python/engine_runtime_profile.py`
- Rust parity contract: `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`
- Regression coverage: `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py`

## Enabled behavior

- The runtime profile can now describe whether a deployment supports llama-swap-style routing semantics.
- Consumers can use the profile to select policy, emit telemetry, or gate future routing work.

## Disabled behavior

- All routing capability flags default to `false`.
- No gateway or request-router behavior is introduced by this wave.

## Fallback behavior

- If a consumer does not recognize these fields, the profile still serializes cleanly through existing `to_dict` / serde paths.
- Unsupported routing behavior remains opt-in and fail-closed.

## Why this donor matters

- `llama-swap` is the cleanest control-plane donor in the remaining set.
- Its features map well to gfxATOM policy metadata without forcing a full serving gateway into the repo.
- This makes it a low-risk assimilation step before larger routing or model-packing work.
