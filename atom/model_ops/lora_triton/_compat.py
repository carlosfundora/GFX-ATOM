# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — provides vllm-like utilities without a vllm dependency.

This module is imported by the LoRA FP8 Triton kernels ported from vLLM.
It replaces `vllm.triton_utils`, `vllm.utils.torch_utils`,
`vllm.utils.math_utils`, `vllm.logger`, and `vllm.platforms`.
"""

import logging
import math
import os
import functools
from typing import Any, Callable

import torch

# ---------- triton re-exports (vllm.triton_utils) ----------
import triton
import triton.language as tl

# ---------- logger (vllm.logger) ----------
def init_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

# ---------- math utils (vllm.utils.math_utils) ----------
def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()

# ---------- torch_utils (vllm.utils.torch_utils) ----------
def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str] | tuple[str, ...] = (),
    fake_impl: Callable | None = None,
    dispatch_key: str = "CUDA",
    tags: tuple = (),
    **kwargs: Any,
):
    """Register a PyTorch custom op. Thin wrapper over torch.library.

    In ATOM we skip custom op registration for LoRA kernels and call them
    as regular Python functions instead, since ATOM does not use
    torch.compile on LoRA paths.
    """
    pass  # no-op: direct Python calls are fine for these Triton kernels

# ---------- platforms (vllm.platforms) ----------
class _Platform:
    """Stub for vllm.platforms.current_platform."""
    _is_rocm: bool | None = None

    @staticmethod
    def is_rocm() -> bool:
        if _Platform._is_rocm is None:
            _Platform._is_rocm = torch.version.hip is not None
        return _Platform._is_rocm

    @staticmethod
    def is_cuda() -> bool:
        return not _Platform.is_rocm()

    @staticmethod
    def get_device_capability() -> tuple[int, int]:
        if _Platform.is_rocm():
            return (9, 0)  # MI300X-class
        props = torch.cuda.get_device_properties(0)
        return (props.major, props.minor)

current_platform = _Platform()

# ---------- distributed (vllm.distributed) ----------
def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Stub — returns input unchanged for single-GPU LoRA paths."""
    return input_

def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """Stub — returns input unchanged for single-GPU LoRA paths."""
    return input_
