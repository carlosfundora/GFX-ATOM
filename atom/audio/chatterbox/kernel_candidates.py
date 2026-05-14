# SPDX-License-Identifier: Apache-2.0
"""Local donor-kernel classification for Chatterbox backend tuning.

This module documents which local kernel trees are viable for Chatterbox T3 on
RDNA2 and which are intentionally kept as references only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelCandidate:
    name: str
    path: Path
    classification: str
    chatterbox_use: str
    default_enabled: bool = False

    @property
    def exists(self) -> bool:
        return self.path.exists()


_ROOT = Path("/home/local/ai")


KERNEL_CANDIDATES: tuple[KernelCandidate, ...] = (
    KernelCandidate(
        name="vllm-rdna2-index-fallback",
        path=_ROOT / "build/kernels/vllm-rdna2",
        classification="RDNA2 vLLM crash workaround",
        chatterbox_use="Safe to adapt for GPU-side indexing failures only.",
    ),
    KernelCandidate(
        name="atom-rdna2-ops",
        path=Path(__file__).resolve().parents[2] / "model_ops/rdna2",
        classification="Existing ATOM RDNA2 operation kernels",
        chatterbox_use="Probe RMSNorm/RoPE/topk/softmax helpers before any T3 wiring.",
    ),
    KernelCandidate(
        name="aiter-triton-gfx1030",
        path=_ROOT / "build/kernels/aiter-triton-gfx1030",
        classification="gfx1030-validated Triton attention/topk/softmax donor",
        chatterbox_use=(
            "Attention/topk/softmax donor only; GEMM remains opt-in after "
            "shape-specific microbench validation."
        ),
    ),
    KernelCandidate(
        name="deepspeed-hip-linear",
        path=_ROOT / "build/kernels/deepspeed-hip/hip_linear",
        classification="Experimental quantized FP6 HIP linear",
        chatterbox_use=(
            "Reference only for Chatterbox FP16 T3 until weight packing and "
            "runtime contract are proven compatible."
        ),
    ),
    KernelCandidate(
        name="llama-cpp-tq3-kvcache",
        path=_ROOT / "build/kernels/llama-cpp-tq3-kvcache",
        classification="GGUF/ggml quantized KV-cache patch",
        chatterbox_use="Not a direct fix for vLLM/transformer Chatterbox T3.",
    ),
    KernelCandidate(
        name="sglang-prism-q1",
        path=_ROOT / "build/kernels/sglang-prism-q1",
        classification="SGLang/GGUF one-bit quantization patch",
        chatterbox_use="Not a direct fix for vLLM/transformer Chatterbox T3.",
    ),
)


def donor_kernel_status() -> list[dict[str, object]]:
    """Return a stable, testable summary of local donor kernel candidates."""
    return [
        {
            "name": candidate.name,
            "path": str(candidate.path),
            "exists": candidate.exists,
            "classification": candidate.classification,
            "chatterbox_use": candidate.chatterbox_use,
            "default_enabled": candidate.default_enabled,
        }
        for candidate in KERNEL_CANDIDATES
    ]


def rdna2_runtime_detected() -> bool:
    """Best-effort RDNA2/gfx1030 detection without requiring torch import."""
    override = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
    if override.startswith("10.3"):
        return True
    target = os.environ.get("PYTORCH_ROCM_ARCH") or os.environ.get("AMDGPU_TARGETS", "")
    return any(arch in target for arch in ("gfx1030", "gfx1031"))


def allow_experimental_chatterbox_gemm() -> bool:
    """Opt-in gate for donor GEMM kernels that are not default-safe on RDNA2."""
    return os.environ.get("ATOM_CHATTERBOX_EXPERIMENTAL_GEMM", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


__all__ = [
    "KERNEL_CANDIDATES",
    "KernelCandidate",
    "allow_experimental_chatterbox_gemm",
    "donor_kernel_status",
    "rdna2_runtime_detected",
]
