# Wave 12 vLLM FP8 KV cache capability profile

## Source donor

- `gfxATOM-Rust/donors/vllm`

## What was assimilated

- FP8 KV cache support as a capability surface.
- Per-tensor and per-head scale support.
- Calibration-aware KV scale generation.
- Explicit non-fused quantized attention state.

## Integration path

- Python contract: `gfxATOM-Rust/python/engine_runtime_profile.py`
- Rust parity contract: `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`
- Regression coverage: `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py`

## Enabled behavior

- The runtime profile can now describe the exact FP8 KV-cache shape a deployment supports.
- Consumers can distinguish plain FP8 cache support from calibrated, head-scaled, or future fused attention variants.

## Disabled behavior

- All FP8 KV-cache detail flags default to `false`.
- No fused attention kernel or calibration pipeline is introduced by this wave.

## Fallback behavior

- Consumers that ignore the new fields still deserialize the profile cleanly.
- If quantized attention fusion is unavailable, the runtime profile explicitly stays in the non-fused lane.

## Why this donor matters

- The vLLM KV-cache docs provide the clearest practical FP8 capability matrix in the remaining donor set.
- It fills in the detail behind the existing coarse `supports_fp8_kv_cache` flag.
- That keeps gfxATOM's runtime contract precise without committing to a backend implementation yet.
