# Wave 32: FP8 KV Cache Dimension Alignment Contract

## Summary

Added explicit FP8 KV cache dimension alignment validation and constraint support to `rs_kv_quant_contracts`. This addresses a critical correctness issue in upstream ATOM's DeepSeek v2 implementation and ensures that gfxATOM-Rust's KV policy layer maintains alignment guarantees for vectorized FP8 access patterns.

## Problem Statement

FP8 quantized KV caches require 16-byte aligned dimensions for efficient vectorized operations. Upstream ATOM's DeepSeek v2 model computes `head_dim + 4` (e.g., 128 + 4 = 132), which is not aligned to 16 bytes. This causes memory layout violations and can lead to silent correctness bugs during inference.

**Upstream fix reference:** commit 10fba75 in carlosfundora/gfxATOM

## Solution

Added to `rs_kv_quant_contracts/src/lib.rs`:

1. **Error variant:** `KvCodecError::Fp8DimensionMisaligned(usize, String)`
   - Explicit error for FP8 dimension misalignment
   - Reports the invalid dimension and the required aligned dimension

2. **Validation function:** `validate_fp8_kv_dimension(head_dim, model_name) -> Result<usize, KvCodecError>`
   - Checks if dimension is divisible by 16
   - Returns error with alignment suggestion if misaligned
   - Can be called by runtime/policy layers before KV allocation

3. **Helper function:** `align_dimension_to_16(dimension) -> usize`
   - Pure function to compute proper aligned dimension
   - Formula: `((dimension + 15) / 16) * 16`
   - Useful for model implementations (e.g., DeepSeek v2: `align_dimension_to_16(head_dim + 4)`)

## Test Coverage

Added two test functions:

- `fp8_kv_dimension_alignment_validation()`: Tests validation logic for aligned and misaligned cases
- `align_dimension_to_16_helper()`: Tests alignment computation including DeepSeek v2 scenario

All tests passing: ✓ 10/10 in `rs_kv_quant_contracts`

## Integration Notes

### gfxATOM-Rust responsibility
- Policy layer validates FP8 dimension alignment before recommending FP8 KV codec
- Added to KV contract layer for portable validation across Rust and Python adapters

### Upstream ATOM responsibility
- Model implementations must respect alignment constraints (or use `align_dimension_to_16()` helper pattern)
- KV connector initialization must validate dimensions during warmup phase
- Ref: upstream commit d674248 for KV warmup hook

## Downstream Use

The validation contract is available for:
1. **Engine profile layer** (`python/engine_runtime_profile.py`): Can validate FP8 KV support
2. **KV policy arbiters** (`python/kv_policy_arbiter.py`): Can enforce alignment before policy recommendation
3. **Runtime integrations**: Can call helper before allocating FP8 KV blocks

## Compatibility

- Backward compatible: new error variant and functions do not affect existing contract consumers
- KvCodecError now derives Clone to support idiomatic error handling

## Related Issues

- **Upstream ATOM:** DeepSeek v2 FP8 KV misalignment (10fba75)
- **Upstream ATOM:** KV connector warmup initialization (d674248)
- **Upstream ATOM:** Security: pickle deserialization (887788d)

## Next Steps

1. Verify that upstream ATOM applies commit 10fba75 for DeepSeek v2
2. Ensure KV warmup initialization hook is in place (d674248)
3. Document in engine profile which models require FP8 alignment validation
