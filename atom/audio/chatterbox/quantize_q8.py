# SPDX-License-Identifier: Apache-2.0
"""CLI for creating Chatterbox ONNX Q8 sidecar artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from atom.audio.chatterbox.onnx_artifacts import Q8_COMPONENTS, quantize_chatterbox_q8_sidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_model_dir", type=Path)
    parser.add_argument("output_model_dir", type=Path)
    parser.add_argument(
        "--component",
        action="append",
        choices=Q8_COMPONENTS,
        help="Component to quantize. May be repeated. Defaults to all Chatterbox ONNX components.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = quantize_chatterbox_q8_sidecar(
        args.source_model_dir,
        args.output_model_dir,
        components=args.component or Q8_COMPONENTS,
        overwrite=args.overwrite,
    )
    for result in results:
        action = "quantized" if result.quantized else "reused"
        print(f"{action}: {result.component} -> {result.output_path}")


if __name__ == "__main__":
    main()
