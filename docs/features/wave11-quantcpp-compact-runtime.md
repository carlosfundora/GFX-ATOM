# Wave 11 quant.cpp compact runtime capability profile

## Source donor

- `.archived/repos/quant.cpp`

## What was assimilated

- CPU-only / embeddable runtime semantics.
- On-demand model download and local model cache behavior.
- Ollama-style CLI plus OpenAI-compatible server capability.
- Progressive KV compression and full-document mode as compact-runtime signals.

## Integration path

- Python contract: `gfxATOM-Rust/python/engine_runtime_profile.py`
- Rust parity contract: `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`
- Regression coverage: `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py`

## Enabled behavior

- The runtime profile can now describe whether a deployment behaves like a compact local quant.cpp-style runtime.
- Consumers can use the profile to decide whether to route to a CPU-only, download-on-demand, document-first mode.

## Disabled behavior

- All compact-runtime flags default to `false`.
- No CLI or server implementation is introduced by this wave.

## Fallback behavior

- Consumers that ignore the new fields still deserialize the runtime profile normally.
- Unsupported compact-runtime behavior remains opt-in and fail-closed.

## Why this donor matters

- `quant.cpp` is the leanest remaining runtime donor.
- Its unique value is not routing or packing, but a compact local runtime with progressive KV compression and document-wide execution.
- This gives gfxATOM a precise way to describe CPU/embeddable local-serving behavior without adding another server stack.
