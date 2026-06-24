#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gguf_pipeline_comparator import compare_engines, default_engine_paths, write_report_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare GGUF handling across SGLang, ATOM, and llama.cpp.")
    parser.add_argument("--sglang-path", type=Path, default=default_engine_paths()["sglang"])
    parser.add_argument("--atom-path", type=Path, default=default_engine_paths()["atom"])
    parser.add_argument("--llama-path", type=Path, default=default_engine_paths()["llama.cpp"])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "gguf_pipeline_comparison.json",
        help="Output JSON report path.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = compare_engines(
        {
            "sglang": args.sglang_path,
            "atom": args.atom_path,
            "llama.cpp": args.llama_path,
        }
    )
    write_report_json(report, args.output)

    print("GGUF pipeline comparison complete")
    print(f"Ranking: {' > '.join(report.ranking)}")
    print(f"Report:  {args.output}")
    print("Top assimilation steps:")
    for step in report.assimilation_steps:
        print(f"  [{step.priority}] {step.step_id} <- {step.source_engine} ({step.target_module})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
