# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for Chatterbox ONNX Runtime helpers."""

from atom.audio.runtime import (
    DEFAULT_ONNX_CPU_PROVIDER,
    DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL,
    OnnxCpuRuntimeConfig,
    create_cpu_inference_session,
    create_cpu_session_options,
    create_onnx_cpu_inference_session,
    create_onnx_cpu_session_options,
    physical_core_count,
)

__all__ = [
    "DEFAULT_ONNX_CPU_PROVIDER",
    "DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL",
    "OnnxCpuRuntimeConfig",
    "create_cpu_inference_session",
    "create_cpu_session_options",
    "create_onnx_cpu_inference_session",
    "create_onnx_cpu_session_options",
    "physical_core_count",
]
