"""GGUF quantized weight operations for ATOM.

Provides GPU-accelerated dequantization, matrix-vector multiply, and
fused MoE dispatch for GGUF-quantized weights (Q1_0 through Q8_0,
K-quants, I-matrix quants, and PRISM Q1).

GPU kernels come from sgl_kernel (compiled C++/HIP .so). Falls back to
CPU dequantization via the ``gguf`` Python library when GPU kernels are
unavailable.

Usage:
    from atom.quantization.gguf_ops import gguf_matmul, gguf_dequantize

    # Decode-time hot path: fused quantized GEMV
    output = gguf_matmul(activations, qweight, qweight_type)

    # Full dequant for prefill/batch > threshold
    weight_fp16 = gguf_dequantize(qweight, qweight_type, dtype=torch.float16)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import torch

from atom.quantization.gguf_compat import (
    PRISM_Q1_0,
    PRISM_Q1_0_G128,
    ensure_prism_gguf_compat,
    gguf_type_name,
)

logger = logging.getLogger(__name__)

# Register PRISM types with gguf library at import time
ensure_prism_gguf_compat()

# Detect GPU backend
_has_gpu_kernels = False
_is_hip = hasattr(torch.version, "hip") if hasattr(torch, "version") else False

try:
    from sgl_kernel import gelu_and_mul, silu_and_mul
    from sgl_kernel.quantization import (
        ggml_dequantize,
        ggml_moe_a8,
        ggml_moe_a8_vec,
        ggml_moe_get_block_size,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    _has_gpu_kernels = True
    logger.info("GGUF ops: sgl_kernel GPU kernels available")
except ImportError:
    warnings.warn(
        "sgl_kernel not found — GGUF quantization will use CPU dequantize fallback. "
        "Install sgl_kernel for GPU-accelerated GGUF inference."
    )

try:
    from sgl_kernel import moe_align_block_size, moe_sum
except ImportError:
    moe_align_block_size = None
    moe_sum = None

# ── Weight type sets ────────────────────────────────────────────────

try:
    from gguf import GGMLQuantizationType as WeightType

    UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}
    STANDARD_QUANT_TYPES = {
        WeightType.Q4_0, WeightType.Q4_1,
        WeightType.Q5_0, WeightType.Q5_1,
        WeightType.Q8_0, WeightType.Q8_1,
    }
    KQUANT_TYPES = {
        WeightType.Q2_K, WeightType.Q3_K, WeightType.Q4_K,
        WeightType.Q5_K, WeightType.Q6_K,
    }
    IMATRIX_QUANT_TYPES = {
        WeightType.IQ1_M, WeightType.IQ1_S,
        WeightType.IQ2_XXS, WeightType.IQ2_XS, WeightType.IQ2_S,
        WeightType.IQ3_XXS, WeightType.IQ3_S,
        WeightType.IQ4_XS, WeightType.IQ4_NL,
    }
except ImportError:
    # gguf not installed — define empty sets so the module can still be imported
    UNQUANTIZED_TYPES = set()
    STANDARD_QUANT_TYPES = set()
    KQUANT_TYPES = set()
    IMATRIX_QUANT_TYPES = set()

PRISM_Q1_TYPES = {PRISM_Q1_0, PRISM_Q1_0_G128}
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES | PRISM_Q1_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES


# ── Core operations ────────────────────────────────────────────────

def gguf_dequantize(
    qweight: torch.Tensor,
    qweight_type: int,
    *,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize a GGUF-quantized weight tensor to the given dtype.

    Uses GPU kernels when available, falls back to CPU via gguf library.

    Args:
        qweight: Quantized weight tensor [rows, packed_cols]
        qweight_type: GGML quantization type ID
        dtype: Output dtype (float16, bfloat16, or float32)

    Returns:
        Dequantized weight tensor [rows, cols]
    """
    import gguf

    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    rows = qweight.shape[0]
    cols = qweight.shape[1] // type_size * block_size

    # GPU path
    if _has_gpu_kernels and qweight.is_cuda:
        return ggml_dequantize(qweight, qweight_type, rows, cols, dtype)

    # CPU fallback
    import gguf.quants as gguf_quants
    import numpy as np

    qw_np = qweight.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    dequant = gguf_quants.dequantize(qw_np, qweight_type)
    dequant = np.ascontiguousarray(dequant.reshape(rows, cols))
    return torch.from_numpy(dequant).to(device=qweight.device, dtype=dtype)


def gguf_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
) -> torch.Tensor:
    """Quantized matrix multiplication: x @ dequant(qweight).T

    Dispatches to the fastest available kernel:
      1. MMVQ (fused quantized GEMV) for small batches
      2. MMQ (block-quantized GEMM) for medium batches
      3. GPU dequant + cuBLAS for large batches

    Args:
        x: Activation tensor [batch, hidden]
        qweight: Quantized weight [out_features, packed_in_features]
        qweight_type: GGML quantization type ID

    Returns:
        Output tensor [batch, out_features]
    """
    if x.shape[0] == 0:
        return torch.empty(0, qweight.shape[0], dtype=x.dtype, device=x.device)
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T

    # MMVQ batch thresholds
    if qweight_type in PRISM_Q1_TYPES:
        mmvq_safe = 8
    elif qweight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8
    else:
        mmvq_safe = 2 if qweight.shape[0] > 5120 else 6

    # Tier 1: MMVQ kernel (decode hot path)
    if _has_gpu_kernels and x.shape[0] <= mmvq_safe and (
        qweight_type in MMVQ_QUANT_TYPES or qweight_type in PRISM_Q1_TYPES
    ):
        return ggml_mul_mat_vec_a8(
            qweight, x.contiguous(), qweight_type, qweight.shape[0]
        )

    # Tier 2: MMQ kernel
    if _has_gpu_kernels and qweight_type in MMQ_QUANT_TYPES:
        return ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])

    # Tier 3: GPU dequant + matmul
    if qweight_type in DEQUANT_TYPES:
        weight = gguf_dequantize(qweight, qweight_type, dtype=x.dtype)
        return x @ weight.T

    raise NotImplementedError(
        f"Unsupported GGUF quantization type: {gguf_type_name(qweight_type)}"
    )


def gguf_fused_moe(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str = "silu",
) -> torch.Tensor:
    """Fused MoE with GGUF-quantized expert weights.

    Args:
        x: Input hidden states [num_tokens, hidden_size]
        w1: Gate+up expert weights [num_experts, 2*intermediate, packed_hidden]
        w2: Down expert weights [num_experts, hidden, packed_intermediate]
        topk_weights: Expert weights [num_tokens, top_k]
        topk_ids: Expert indices [num_tokens, top_k]
        qweight_type: GGML type for w1
        qweight_type2: GGML type for w2
        activation: "silu" or "gelu"

    Returns:
        Output hidden states [num_tokens, hidden_size]
    """

    def act(x: torch.Tensor) -> torch.Tensor:
        if _has_gpu_kernels:
            d = x.shape[-1] // 2
            out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
            if activation == "silu":
                silu_and_mul(out, x)
            elif activation == "gelu":
                gelu_and_mul(out, x)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            return out
        gate, up = x.chunk(2, dim=-1)
        if activation == "silu":
            return torch.nn.functional.silu(gate) * up
        if activation == "gelu":
            return torch.nn.functional.gelu(gate) * up
        raise ValueError(f"Unsupported activation: {activation}")

    out_hidden_states = torch.empty_like(x)

    # Fast path: sgl_kernel MoE kernels
    if (
        _has_gpu_kernels
        and moe_align_block_size is not None
        and qweight_type2 in MMQ_QUANT_TYPES
        and qweight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        BLOCK_SIZE = ggml_moe_get_block_size(qweight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, BLOCK_SIZE, E
        )
        out = ggml_moe_a8(
            x, w1, sorted_token_ids, expert_ids, num_tokens_post_padded,
            qweight_type, N, top_k, num_tokens,
        )
        out = act(out)
        out = ggml_moe_a8(
            out, w2, sorted_token_ids, expert_ids, num_tokens_post_padded,
            qweight_type2, w2.shape[1], 1, num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        moe_sum(out, out_hidden_states)

    elif (
        _has_gpu_kernels
        and qweight_type2 in MMVQ_QUANT_TYPES
        and qweight_type in MMVQ_QUANT_TYPES
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]

        out = ggml_moe_a8_vec(x, w1, topk_ids, top_k, qweight_type, N, num_tokens)
        out = act(out)
        out = ggml_moe_a8_vec(
            out, w2, topk_ids, 1, qweight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        moe_sum(out, out_hidden_states)

    else:
        # Fallback: per-expert sequential dispatch
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape(1, -1)
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                out = gguf_matmul(inp, w1[ii], qweight_type)
                out = act(out)
                current_state = gguf_matmul(out, w2[ii], qweight_type2).mul_(ww)
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state

    return out_hidden_states


def gguf_embedding(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Embedding lookup from GGUF-quantized weight table.

    Args:
        x: Token indices tensor
        qweight: Quantized embedding table [vocab_size, packed_hidden]
        qweight_type: GGML quantization type
        hidden_size: Embedding dimension
        dtype: Output dtype

    Returns:
        Embedded tensor [*x.shape, hidden_size]
    """
    if qweight_type in UNQUANTIZED_TYPES:
        return torch.embedding(qweight, x)
    elif qweight_type in DEQUANT_TYPES:
        x_flat = x.flatten()
        quant = torch.index_select(qweight, dim=0, index=x_flat)
        dequant = gguf_dequantize(quant, qweight_type, dtype=dtype or torch.float16)
        return dequant.view(*x.shape, hidden_size)
    else:
        raise NotImplementedError(
            f"Unsupported GGUF quantization type: {gguf_type_name(qweight_type)}"
        )
