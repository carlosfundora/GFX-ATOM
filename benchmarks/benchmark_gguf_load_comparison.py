from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "debug" / "gguf-benchmark"


def main() -> int:
    if BIN.exists():
        command = [str(BIN), *sys.argv[1:]]
    else:
        command = [
            "cargo",
            "run",
            "-q",
            "-p",
            "rs_gguf_loader_core",
            "--bin",
            "gguf-benchmark",
            "--",
            *sys.argv[1:],
        ]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
