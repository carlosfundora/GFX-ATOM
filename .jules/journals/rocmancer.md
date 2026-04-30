# Rocmancer Journal

## Run 1: Enable Triton MoE for gfx1030

- **What was changed**: Added `gfx.startswith("gfx103")` to the `self.use_triton` assignment condition in `atom/model_ops/moe.py`.
- **Why it matters on gfx1030**: RDNA2 consumer cards previously fell back to slower non-Triton paths. Enabling Triton kernels opens up performance opportunities for MoE models on this architecture.
- **What was learned**: The codebase relies heavily on the `get_gfx()` from `aiter.jit.utils.chip_info` to route hardware-specific kernel logic. Testing logic required mocking specific modules like `aiter` as the local test environment didn't have access to the AMD-specific binaries.
- **What remains risky**: Not explicitly tested on hardware, so Triton kernel availability/compatibility needs field verification. The underlying `fused_moe_triton.py` might still have implicit warp/wavefront size assumptions that differ from `gfx94`.
- **Next steps**: Explore `fused_moe_triton.py` and `attention_mla.py` for wavefront (32 vs 64) or LDS tuning specifically for `gfx1030`, or refine KV cache allocations.