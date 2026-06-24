from __future__ import annotations

from dataclasses import asdict, dataclass

from kv_policy_arbiter import KvPolicyFamily


@dataclass(frozen=True)
class RuntimeSignals:
    kv_hit_rate: float = 0.0
    prefix_reuse_ratio: float = 0.0
    prefill_tokens_per_second: float = 0.0
    decode_tokens_per_second: float = 0.0
    outlier_ratio: float = 0.0
    storage_tier: str = "hbm"


@dataclass(frozen=True)
class AdaptivePolicyRecommendation:
    family: KvPolicyFamily
    score: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["family"] = self.family.value
        return payload


def recommend_policy_family(signals: RuntimeSignals) -> AdaptivePolicyRecommendation:
    # Start from baseline and move to quant-aware families only when runtime signals
    # indicate concrete value.
    family = KvPolicyFamily.baseline
    score = 0.5
    reason = "baseline_default"

    if signals.prefix_reuse_ratio >= 0.65 and signals.kv_hit_rate >= 0.55:
        family = KvPolicyFamily.ratequant
        score = 0.82
        reason = "high_prefix_reuse_high_kv_hit"
    elif signals.outlier_ratio >= 0.14:
        family = KvPolicyFamily.nautilus
        score = 0.78
        reason = "high_outlier_ratio_guarded_quant"
    elif signals.storage_tier in {"cpu", "nvme", "object"} and signals.kv_hit_rate < 0.45:
        family = KvPolicyFamily.deltak
        score = 0.74
        reason = "cold_storage_low_kv_hit_delta_friendly"

    if signals.decode_tokens_per_second > 0 and signals.prefill_tokens_per_second > 0:
        decode_ratio = signals.decode_tokens_per_second / max(1.0, signals.prefill_tokens_per_second)
        if family is KvPolicyFamily.baseline and decode_ratio < 0.35:
            family = KvPolicyFamily.wobble
            score = 0.69
            reason = "decode_skew_recommend_wobble"

    return AdaptivePolicyRecommendation(family=family, score=score, reason=reason)
