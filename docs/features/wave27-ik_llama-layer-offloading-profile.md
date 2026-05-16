# Wave-27 ik_llama.cpp layer-offloading runtime profile

## Source donor extraction

- Donor: `ik_llama.cpp`
- Extracted implementation idea:
  - CPU/hybrid layer offloading and multi-GPU placement for llama.cpp-style runtimes

## Assimilation target

- `gfxATOM-Rust/python/engine_runtime_profile.py`
- `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`

## Runtime gating

- This is a capability-profile lane rather than a hard runtime toggle.
- Consumers can advertise the offload capability explicitly through the runtime profile.

## Behavior

- Adds a dedicated layer-offloading capability flag to the runtime profile surface.
- Keeps the profile contract aligned between Python and Rust.
- Makes CPU/hybrid placement support visible without changing execution behavior.

## Fallback behavior

- The capability defaults to disabled.
- Existing runtime profiles remain unchanged unless the new helper is used.

## Why this donor matters

- `ik_llama.cpp` is a useful donor for CPU/hybrid runtime and offload strategy ideas.
- The profile flag gives the orchestrator a clean place to advertise offload support later.
- The result is a minimal, durable surface for future offload benchmarking and routing work.
