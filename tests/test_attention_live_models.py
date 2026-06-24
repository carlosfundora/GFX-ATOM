#!/usr/bin/env python3
"""
Real GGUF model-load smoke suite (non-simulated).

This replaces the prior simulated attention test harness with real model loads
through llama.cpp so regressions in GGUF compatibility are caught early.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Iterable

import pytest


MODELS_ROOT = Path("/home/local/ai/models")
LLAMA_CLI = Path("/home/local/ai/projects/donors/llama.cpp-1-bit-turbo/build/bin/llama-cli")
ROOT = Path(__file__).resolve().parents[1]
QUANT_RE = re.compile(r"\.(Q[0-9_]+[^.]*)\.gguf$", re.IGNORECASE)


@dataclass
class LoadSmokeResult:
    quantization: str
    model_path: str
    ok: bool
    return_code: int
    elapsed_ms: float
    output_tail: str


def _discover_gguf_quant_samples(models_root: Path) -> dict[str, Path]:
    ggufs = list(models_root.rglob("*.gguf"))
    buckets: dict[str, list[Path]] = {}
    for model in ggufs:
        match = QUANT_RE.search(model.name)
        quant = match.group(1).upper() if match else "UNKNOWN"
        buckets.setdefault(quant, []).append(model)

    # Use smallest model in each quant bucket to keep runtime bounded.
    return {quant: min(paths, key=lambda p: p.stat().st_size) for quant, paths in buckets.items()}


def _run_llama_cli_load(model_path: Path, timeout_seconds: int = 180) -> LoadSmokeResult:
    command = [
        str(LLAMA_CLI),
        "--model",
        str(model_path),
        "--threads",
        "4",
        "--ctx-size",
        "128",
        "--predict",
        "1",
        "--prompt",
        "hello",
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
        "--no-warmup",
        "--device",
        "none",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        output = completed.stdout
        ok = completed.returncode == 0 and "error:" not in output.lower() and "failed" not in output.lower()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return LoadSmokeResult(
            quantization=(QUANT_RE.search(model_path.name).group(1).upper() if QUANT_RE.search(model_path.name) else "UNKNOWN"),
            model_path=str(model_path),
            ok=ok,
            return_code=completed.returncode,
            elapsed_ms=elapsed_ms,
            output_tail="\n".join(output.splitlines()[-8:]),
        )
    except subprocess.TimeoutExpired:
        return LoadSmokeResult(
            quantization=(QUANT_RE.search(model_path.name).group(1).upper() if QUANT_RE.search(model_path.name) else "UNKNOWN"),
            model_path=str(model_path),
            ok=False,
            return_code=124,
            elapsed_ms=float(timeout_seconds * 1000),
            output_tail="timeout",
        )


def run_real_load_suite(full_quant_coverage: bool = False) -> list[LoadSmokeResult]:
    quant_samples = _discover_gguf_quant_samples(MODELS_ROOT)
    selected: Iterable[tuple[str, Path]]
    if full_quant_coverage:
        selected = sorted(quant_samples.items())
    else:
        # Fast non-regression subset used by default pytest runs.
        priority = ["Q4_K_M", "Q8_0", "Q2_K", "Q6_K"]
        picked: list[tuple[str, Path]] = []
        for quant in priority:
            if quant in quant_samples:
                picked.append((quant, quant_samples[quant]))
        if not picked:
            picked = sorted(quant_samples.items())[:3]
        selected = picked

    return [_run_llama_cli_load(path) for _, path in selected]


def _preconditions_or_skip() -> None:
    if not LLAMA_CLI.exists():
        pytest.skip(f"llama-cli not available at {LLAMA_CLI}")
    if not MODELS_ROOT.exists():
        pytest.skip(f"models root missing at {MODELS_ROOT}")
    if not list(MODELS_ROOT.rglob("*.gguf")):
        pytest.skip("no .gguf models found under /home/local/ai/models")


def test_real_gguf_load_smoke_non_regression():
    _preconditions_or_skip()
    results = run_real_load_suite(full_quant_coverage=False)
    failures = [result for result in results if not result.ok]
    assert not failures, f"GGUF load failures: {[asdict(f) for f in failures]}"


def test_real_gguf_quantization_bucket_discovery():
    _preconditions_or_skip()
    quant_samples = _discover_gguf_quant_samples(MODELS_ROOT)
    assert len(quant_samples) >= 3


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run real GGUF load smoke tests via llama.cpp")
    parser.add_argument("--full", action="store_true", help="Run one real load per discovered quantization bucket.")
    parser.add_argument(
        "--output",
        default=ROOT / "benchmarks" / "gguf_real_load_results.json",
        type=Path,
    )
    args = parser.parse_args()

    if not LLAMA_CLI.exists() or not MODELS_ROOT.exists():
        print("required paths missing; cannot run real GGUF loads")
        return 2

    results = run_real_load_suite(full_quant_coverage=args.full)
    payload = {
        "total": len(results),
        "passed": sum(1 for row in results if row.ok),
        "failed": sum(1 for row in results if not row.ok),
        "total_load_time_ms": sum(row.elapsed_ms for row in results),
        "results": [asdict(row) for row in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"saved: {args.output}")
    return 0 if payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
