# SPDX-License-Identifier: Apache-2.0
"""Shared runtime helpers for gfxATOM audio engines."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

DEFAULT_ONNX_CPU_PROVIDER = "CPUExecutionProvider"
DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL = "ORT_ENABLE_ALL"


def physical_core_count(default: int = 1) -> int:
    """Return physical CPU cores, falling back to logical cores when needed."""
    try:
        import psutil

        count = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
    except Exception:
        count = os.cpu_count()
    return max(int(count or default), 1)


@dataclass(frozen=True)
class OnnxCpuRuntimeConfig:
    """ONNX Runtime CPU session settings for audio inference components."""

    num_threads: Optional[int] = None
    inter_op_threads: int = 1
    graph_optimization_level: str = DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL
    providers: Sequence[str] = (DEFAULT_ONNX_CPU_PROVIDER,)

    @property
    def intra_op_threads(self) -> int:
        return max(int(self.num_threads or physical_core_count()), 1)

    def create_session_options(self):
        return create_onnx_cpu_session_options(
            num_threads=self.num_threads,
            inter_op_threads=self.inter_op_threads,
            graph_optimization_level=self.graph_optimization_level,
        )

    def create_inference_session(self, model_path: str):
        return create_onnx_cpu_inference_session(
            model_path,
            num_threads=self.num_threads,
            inter_op_threads=self.inter_op_threads,
            graph_optimization_level=self.graph_optimization_level,
            providers=self.providers,
        )


def create_onnx_cpu_session_options(
    num_threads: Optional[int] = None,
    inter_op_threads: int = 1,
    graph_optimization_level: str = DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL,
):
    """Create ONNX Runtime CPU session options tuned for one audio request."""
    import onnxruntime

    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = max(int(inter_op_threads), 1)
    opts.intra_op_num_threads = max(int(num_threads or physical_core_count()), 1)
    opts.graph_optimization_level = getattr(
        onnxruntime.GraphOptimizationLevel,
        graph_optimization_level,
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
    )
    return opts


def create_onnx_cpu_inference_session(
    model_path: str,
    *,
    num_threads: Optional[int] = None,
    inter_op_threads: int = 1,
    graph_optimization_level: str = DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL,
    providers: Sequence[str] = (DEFAULT_ONNX_CPU_PROVIDER,),
):
    """Load an ONNX Runtime session using CPU-oriented audio defaults."""
    import onnxruntime

    return onnxruntime.InferenceSession(
        model_path,
        create_onnx_cpu_session_options(
            num_threads=num_threads,
            inter_op_threads=inter_op_threads,
            graph_optimization_level=graph_optimization_level,
        ),
        providers=list(providers),
    )


create_cpu_session_options = create_onnx_cpu_session_options
create_cpu_inference_session = create_onnx_cpu_inference_session

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
