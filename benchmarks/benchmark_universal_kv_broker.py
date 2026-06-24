#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sglang_backend_adapter import TieredKvCacheAdapter  # noqa: E402


MODEL_SHAPES = {
    "lfm": {"heads": 8, "head_dim": 16, "seq_len": 96},
    "qwen": {"heads": 8, "head_dim": 16, "seq_len": 96},
    "opencoder": {"heads": 8, "head_dim": 16, "seq_len": 128},
}


@dataclass
class UniversalKvBenchmarkResult:
    models: list[str]
    gpu_capacity_mb: int
    ram_capacity_mb: int
    allocated_blocks: int
    vram_hot_hit_rate: float
    spill_restore_latency_ms_p50: float
    spill_restore_latency_ms_p95: float
    effective_context_expansion: float


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, int(round((len(sorted_vals) - 1) * q))))
    return float(sorted_vals[idx])


def run_benchmark(models: list[str], gpu_capacity_mb: int = 8, ram_capacity_mb: int = 64) -> UniversalKvBenchmarkResult:
    adapter = TieredKvCacheAdapter(
        gpu_capacity_mb=gpu_capacity_mb,
        ram_capacity_mb=ram_capacity_mb,
        dimension=128,
        num_heads=8,
    )

    block_ids: list[int] = []
    for cycle in range(12):
        model = models[cycle % len(models)]
        shape = MODEL_SHAPES[model]
        features = shape["heads"] * shape["head_dim"]
        seq_len = shape["seq_len"]
        k_cache = torch.randn(seq_len, features, dtype=torch.float32)
        v_cache = torch.randn(seq_len, features, dtype=torch.float32)
        importance = 0.9 if cycle % 3 == 0 else 0.25
        block_id = adapter.allocate_kv_block(
            request_id=f"{model}-{cycle}",
            layer_idx=cycle % 8,
            k_cache=k_cache,
            v_cache=v_cache,
            importance_score=importance,
        )
        block_ids.append(block_id)

    latency_ms: list[float] = []
    # Access older blocks first to sample spill/restore behavior.
    for block_id in block_ids[: len(block_ids) // 2]:
        t0 = time.perf_counter()
        adapter.get_kv_block(block_id)
        latency_ms.append((time.perf_counter() - t0) * 1000.0)

    stats = adapter.get_cache_stats()
    vram_hot_hit_rate = float(stats["gpu_tier"]["hit_rate"])
    # Approximate expansion from 2-tier compressed pool vs single-tier baseline.
    effective_context_expansion = (gpu_capacity_mb + ram_capacity_mb) / max(1, gpu_capacity_mb)

    return UniversalKvBenchmarkResult(
        models=models,
        gpu_capacity_mb=gpu_capacity_mb,
        ram_capacity_mb=ram_capacity_mb,
        allocated_blocks=len(block_ids),
        vram_hot_hit_rate=vram_hot_hit_rate,
        spill_restore_latency_ms_p50=_percentile(latency_ms, 0.50),
        spill_restore_latency_ms_p95=_percentile(latency_ms, 0.95),
        effective_context_expansion=effective_context_expansion,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal KV Broker benchmark harness.")
    parser.add_argument("--models", default="lfm,qwen,opencoder", help="Comma-separated models.")
    parser.add_argument("--gpu-capacity-mb", type=int, default=8)
    parser.add_argument("--ram-capacity-mb", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "universal_kv_benchmark.json",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    unsupported = [m for m in models if m not in MODEL_SHAPES]
    if unsupported:
        raise ValueError(f"Unsupported model keys: {unsupported}. Supported keys: {sorted(MODEL_SHAPES)}")

    result = run_benchmark(models, gpu_capacity_mb=args.gpu_capacity_mb, ram_capacity_mb=args.ram_capacity_mb)
    payload = asdict(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("Universal KV benchmark complete")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
