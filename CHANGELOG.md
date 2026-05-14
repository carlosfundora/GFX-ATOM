# Changelog

All notable changes to gfxATOM are documented in this file.

## [Unreleased] — 2026-05-13

### Added — Kernel Backend Expansion (Wave 1)

- **Spec Decode HIP Crash Fix**: Added try/except guard with CPU `index_select` fallback in `model_runner.py` for GPU-side fancy indexing that triggers `AcceleratorError` on RDNA2 (gfx1030) during speculative decoding's draft token gathering.

- **SnakeBeta Activation** (`atom/model_ops/rdna2/snake_activation.py`): Ported fused Triton SnakeBeta activation kernel from vLLM-Omni. Used by Qwen3-TTS / Code2Wav audio decoders. Features precomputed exp cache, auto-fallback to eager PyTorch, and inference-mode Triton auto-detection.

- **AITER Triton Flash Attention** (`atom/model_ops/attentions/aiter_triton_flash/`): Integrated the pre-tuned AMD Triton flash attention module with full FA2/FA3 forward and backward interfaces. Includes RDNA2 architecture detection, extended-context autotune configs, and configurable `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON` override.

- **LoRA FP8 Triton Ops** (`atom/model_ops/lora_triton/`): Ported 7 FP8-precision LoRA Triton kernels from vLLM (`lora_shrink_fp8`, `lora_expand_fp8`, `fused_moe_lora_fp8`, `fp8_kernel_utils`, `kernel_utils`, `lora_kernel_metadata`, `utils`). Created `_compat.py` shim to replace all vLLM dependencies (triton_utils, logger, platforms, distributed, torch_utils).

- **Layernorm Gated** (`atom/model_ops/rdna2/layernorm_gated.py`): Ported fused RMSNorm+Gating Triton kernel from vLLM's Mamba ops. Supports grouped normalization with optional gating (SiLU), bias, and both LayerNorm and RMSNorm modes.

- **gfxGRAPH Integration** (`atom/model_ops/gfxgraph.py`): Created ATOM integration layer for the gfxGRAPH Rust bridge (`gfxgraph_rs`). Provides lazy import, `ATOM_ENABLE_GFXGRAPH` environment variable gating, and `create_atom_bucket_router()` factory with standard ATOM batch size buckets. Fixed BucketSelector → BucketRouter alias in `gfxgraph_rs/src/lib.rs`.

- **Environment Variable**: Added `ATOM_ENABLE_GFXGRAPH` to `atom/utils/envs.py` (default: disabled).
