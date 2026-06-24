import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wave1_kv_policy_canary.py"


def test_canary_emits_adaptive_recommendation_when_enabled(tmp_path):
    out_path = tmp_path / "canary.json"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--policy-family",
        "baseline",
        "--adaptive",
        "--prefix-reuse-ratio",
        "0.75",
        "--kv-hit-rate",
        "0.65",
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    adaptive = payload["adaptive_recommendation"]
    assert adaptive is not None
    assert adaptive["family"] == "ratequant"
