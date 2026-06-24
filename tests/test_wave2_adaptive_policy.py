from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from kv_policy_arbiter import KvPolicyFamily  # noqa: E402
from wave2_adaptive_policy import (  # noqa: E402
    RuntimeSignals,
    recommend_policy_family,
)


def test_recommend_ratequant_on_high_prefix_and_hits():
    rec = recommend_policy_family(
        RuntimeSignals(prefix_reuse_ratio=0.8, kv_hit_rate=0.7, storage_tier="hbm")
    )
    assert rec.family == KvPolicyFamily.ratequant
    assert rec.reason == "high_prefix_reuse_high_kv_hit"


def test_recommend_nautilus_on_high_outlier_ratio():
    rec = recommend_policy_family(
        RuntimeSignals(outlier_ratio=0.2, kv_hit_rate=0.5, prefix_reuse_ratio=0.2)
    )
    assert rec.family == KvPolicyFamily.nautilus
    assert rec.reason == "high_outlier_ratio_guarded_quant"


def test_recommend_deltak_for_cold_storage_low_hits():
    rec = recommend_policy_family(
        RuntimeSignals(storage_tier="nvme", kv_hit_rate=0.3, outlier_ratio=0.05)
    )
    assert rec.family == KvPolicyFamily.deltak
    assert rec.reason == "cold_storage_low_kv_hit_delta_friendly"
