# SPDX-License-Identifier: Apache-2.0
# Ported from: vllm/vllm/lora/ops/triton_ops/
# Original Copyright 2024-2025, The vLLM team.
"""FP8 LoRA Triton kernels for ATOM.

Provides fused Triton-based LoRA expand/shrink/MoE operations at FP8 precision.
These are portable across all AMD GPU architectures (MI300X, RDNA2/3) since they
use pure Triton with no HIP or AITER dependencies.

Available operations:
  - lora_shrink_fp8: (x * W_A) with FP8 quantization
  - lora_expand_fp8: (y * W_B) + bias dequantize to FP16
  - fused_moe_lora_fp8: Fused MoE + LoRA at FP8 precision
  - fp8_kernel_utils: Shared FP8 quantize/dequantize helpers
  - kernel_utils: Common Triton kernel launch utilities
  - lora_kernel_metadata: Autotuning metadata for LoRA kernels
"""

from .lora_shrink_fp8_op import lora_shrink_fp8
from .lora_expand_fp8_op import lora_expand_fp8
from .fused_moe_lora_fp8_op import fused_moe_lora_fp8

__all__ = [
    "lora_shrink_fp8",
    "lora_expand_fp8",
    "fused_moe_lora_fp8",
]
