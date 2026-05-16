# Wave 30: vLLM runtime family profile

## Source donor

- `vllm`
- `vllm-omni`
- `vllm-ascend`

## What was assimilated

- `supports_multimodal_serving`
- `supports_omni_modality`
- `supports_hardware_plugin_interface`

## Integration surface

- `gfxATOM-Rust/python/engine_runtime_profile.py`
- `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`
- `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py`
- `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs`

## Behavior

- The new flags only describe runtime capability metadata.
- No execution backend behavior changes with this wave.
- The profile stays fail-closed by default.

## Fallback behavior

- If the vLLM family metadata is absent, the runtime profile remains text-only and backend-agnostic.
- Unsupported adapter capability continues to resolve through existing fallback paths.

## Why it matters

- The vLLM donor family exposes a clean portable seam for multimodal and plugin-backed serving metadata.
- Keeping it in the runtime profile lets the engine negotiate capability shape without binding to a specific backend rewrite.
