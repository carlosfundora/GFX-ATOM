#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent / "python"
sys.path.insert(0, str(PYTHON_DIR))

from kv_policy_arbiter import (  # noqa: E402
    KvPolicyFamily,
    SUPPORTED_FEATURE_FLAGS,
    build_wave1_policy_profile,
)
from wave2_adaptive_policy import (  # noqa: E402
    RuntimeSignals,
    recommend_policy_family,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wave-1 KV policy canary runner with explicit donor feature flags."
    )
    parser.add_argument(
        "--policy-family",
        type=str,
        default="baseline",
        choices=[family.value for family in KvPolicyFamily],
        help="Requested KV policy family.",
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        default="strict",
        choices=["strict", "fallback"],
        help="Policy rejection mode.",
    )
    parser.add_argument(
        "--enable-donor-features",
        action="store_true",
        help="Enable donor feature family selection globally.",
    )
    parser.add_argument(
        "--enable-ratequant",
        action="store_true",
        help="Enable RateQuant policy family.",
    )
    parser.add_argument(
        "--enable-deltak",
        action="store_true",
        help="Enable delta-k policy family.",
    )
    parser.add_argument(
        "--enable-wobble",
        action="store_true",
        help="Enable wobble policy family.",
    )
    parser.add_argument(
        "--enable-qaq",
        action="store_true",
        help="Enable QAQ policy family.",
    )
    parser.add_argument(
        "--enable-nautilus",
        action="store_true",
        help="Enable Nautilus policy family.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/wave1_kv_policy_canary.json"),
        help="Output file for canary decision JSON.",
    )
    parser.add_argument(
        "--total-budget",
        type=float,
        default=32.0,
        help="Total bit budget for ratequant profile simulation.",
    )
    parser.add_argument(
        "--sensitivities",
        type=str,
        default="1.0,1.0,1.0,1.0",
        help="Comma-separated sensitivities for ratequant simulation.",
    )
    parser.add_argument(
        "--vector-dim",
        type=int,
        default=128,
        help="Vector dimension for turboquant profile simulation.",
    )
    parser.add_argument(
        "--bits-per-value",
        type=int,
        default=4,
        help="Requested bits per value before guardrail.",
    )
    parser.add_argument(
        "--outlier-ratio",
        type=float,
        default=0.0,
        help="Outlier ratio for norm-separated guardrail simulation [0,1].",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Emit wave-2 adaptive family recommendation from runtime signals.",
    )
    parser.add_argument("--kv-hit-rate", type=float, default=0.0)
    parser.add_argument("--prefix-reuse-ratio", type=float, default=0.0)
    parser.add_argument("--prefill-tps", type=float, default=0.0)
    parser.add_argument("--decode-tps", type=float, default=0.0)
    parser.add_argument(
        "--storage-tier",
        type=str,
        choices=["hbm", "cpu", "nvme", "object"],
        default="hbm",
    )
    return parser.parse_args()


def feature_overrides_from_args(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "GFXATOM_KV_RATEQUANT": args.enable_ratequant,
        "GFXATOM_KV_DELTAK": args.enable_deltak,
        "GFXATOM_KV_WOBBLE": args.enable_wobble,
        "GFXATOM_KV_QAQ": args.enable_qaq,
        "GFXATOM_KV_NAUTILUS": args.enable_nautilus,
    }


def run() -> None:
    args = parse_args()
    overrides = feature_overrides_from_args(args)
    strict_mode = args.policy_mode == "strict"
    sensitivities = [
        float(chunk.strip())
        for chunk in args.sensitivities.split(",")
        if chunk.strip()
    ]
    profile = build_wave1_policy_profile(
        args.policy_family,
        donor_features_enabled=args.enable_donor_features,
        strict_mode=strict_mode,
        feature_overrides=overrides,
        sensitivities=sensitivities,
        total_budget=args.total_budget,
        vector_dim=args.vector_dim,
        total_bits_per_value=args.bits_per_value,
        outlier_ratio=args.outlier_ratio,
    )
    adaptive = None
    if args.adaptive:
        adaptive = recommend_policy_family(
            RuntimeSignals(
                kv_hit_rate=args.kv_hit_rate,
                prefix_reuse_ratio=args.prefix_reuse_ratio,
                prefill_tokens_per_second=args.prefill_tps,
                decode_tokens_per_second=args.decode_tps,
                outlier_ratio=args.outlier_ratio,
                storage_tier=args.storage_tier,
            )
        ).to_dict()
    payload = {
        "requested_policy_family": args.policy_family,
        "policy_mode": args.policy_mode,
        "donor_features_enabled": args.enable_donor_features,
        "feature_overrides": overrides,
        "profile": profile.to_dict(),
        "adaptive_recommendation": adaptive,
        "supported_feature_flags": {
            family.value: SUPPORTED_FEATURE_FLAGS[family]
            for family in SUPPORTED_FEATURE_FLAGS
        },
        "environment_snapshot": {
            key: os.getenv(key)
            for key in [
                "GFXATOM_DONOR_FEATURES",
                "GFXATOM_KV_POLICY_MODE",
                "GFXATOM_KV_POLICY_FAMILY",
            ]
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wave-1 canary decision written: {args.out}")
    print(json.dumps(payload["profile"], indent=2))


if __name__ == "__main__":
    run()
