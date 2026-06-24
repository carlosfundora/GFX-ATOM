from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from enum import Enum

from wave1_donor_adapters import (
    DeltaKvProfile,
    QaqProfile,
    NautilusProfile,
    RateQuantProfile,
    WobbleProfile,
    TurboQuantProfile,
    NormGuardrailProfile,
    RedundancyAwareKvProfile,
    delta_kv_profile,
    qaq_quality_profile,
    nautilus_geometric_profile,
    ratequant_optimal_bit_allocation,
    wobble_variance_bit_allocation,
    turboquant_profile,
    norm_separated_guardrail,
    rkv_redundancy_profile,
)

class KvPolicyFamily(str, Enum):
    baseline = "baseline"
    ratequant = "ratequant"
    deltak = "deltak"
    wobble = "wobble"
    qaq = "qaq"
    nautilus = "nautilus"
    rkv = "rkv"


SUPPORTED_FEATURE_FLAGS: dict[KvPolicyFamily, str] = {
    KvPolicyFamily.ratequant: "GFXATOM_KV_RATEQUANT",
    KvPolicyFamily.deltak: "GFXATOM_KV_DELTAK",
    KvPolicyFamily.wobble: "GFXATOM_KV_WOBBLE",
    KvPolicyFamily.qaq: "GFXATOM_KV_QAQ",
    KvPolicyFamily.nautilus: "GFXATOM_KV_NAUTILUS",
    KvPolicyFamily.rkv: "GFXATOM_KV_RKV",
}


@dataclass(frozen=True)
class KvPolicyDecision:
    requested_family: str
    selected_family: KvPolicyFamily
    accepted: bool
    rejection_reason: str | None = None
    policy_mode: str = "strict"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Wave1PolicyProfile:
    decision: KvPolicyDecision
    ratequant: RateQuantProfile | None = None
    delta_kv: DeltaKvProfile | None = None
    wobble: WobbleProfile | None = None
    qaq: QaqProfile | None = None
    nautilus: NautilusProfile | None = None
    turboquant: TurboQuantProfile | None = None
    guardrail: NormGuardrailProfile | None = None
    rkv: RedundancyAwareKvProfile | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"decision": self.decision.to_dict()}
        if self.ratequant is not None:
            payload["ratequant"] = self.ratequant.to_dict()
        if self.delta_kv is not None:
            payload["delta_kv"] = self.delta_kv.to_dict()
        if self.wobble is not None:
            payload["wobble"] = self.wobble.to_dict()
        if self.qaq is not None:
            payload["qaq"] = self.qaq.to_dict()
        if self.nautilus is not None:
            payload["nautilus"] = self.nautilus.to_dict()
        if self.turboquant is not None:
            payload["turboquant"] = self.turboquant.to_dict()
        if self.guardrail is not None:
            payload["guardrail"] = self.guardrail.to_dict()
        if self.rkv is not None:
            payload["rkv"] = self.rkv.to_dict()
        return payload


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_policy(
    requested_family: str,
    *,
    donor_features_enabled: bool,
    strict_mode: bool = True,
    feature_overrides: dict[str, bool] | None = None,
) -> KvPolicyDecision:
    requested = requested_family.strip().lower()
    mode = "strict" if strict_mode else "fallback"

    if requested == "":
        requested = KvPolicyFamily.baseline.value

    try:
        family = KvPolicyFamily(requested)
    except ValueError:
        return KvPolicyDecision(
            requested_family=requested_family,
            selected_family=KvPolicyFamily.baseline,
            accepted=False,
            rejection_reason=f"unknown_policy_family:{requested}",
            policy_mode=mode,
        )

    if family is KvPolicyFamily.baseline:
        return KvPolicyDecision(
            requested_family=requested_family,
            selected_family=KvPolicyFamily.baseline,
            accepted=True,
            policy_mode=mode,
        )

    if not donor_features_enabled:
        return KvPolicyDecision(
            requested_family=requested_family,
            selected_family=KvPolicyFamily.baseline,
            accepted=False,
            rejection_reason="donor_features_disabled",
            policy_mode=mode,
        )

    flag_name = SUPPORTED_FEATURE_FLAGS[family]
    flag_enabled = (
        feature_overrides.get(flag_name, False)
        if feature_overrides is not None
        else _env_bool(flag_name, False)
    )

    if flag_enabled:
        return KvPolicyDecision(
            requested_family=requested_family,
            selected_family=family,
            accepted=True,
            policy_mode=mode,
        )

    if strict_mode:
        return KvPolicyDecision(
            requested_family=requested_family,
            selected_family=KvPolicyFamily.baseline,
            accepted=False,
            rejection_reason=f"feature_flag_disabled:{flag_name}",
            policy_mode=mode,
        )

    return KvPolicyDecision(
        requested_family=requested_family,
        selected_family=KvPolicyFamily.baseline,
        accepted=False,
        rejection_reason=f"feature_flag_disabled_fallback:{flag_name}",
        policy_mode=mode,
    )


def resolve_policy_from_env(requested_family: str) -> KvPolicyDecision:
    donor_enabled = _env_bool("GFXATOM_DONOR_FEATURES", False)
    strict_mode = os.getenv("GFXATOM_KV_POLICY_MODE", "strict").strip().lower() != "fallback"
    return resolve_policy(
        requested_family,
        donor_features_enabled=donor_enabled,
        strict_mode=strict_mode,
    )


def build_wave1_policy_profile(
    requested_family: str,
    *,
    donor_features_enabled: bool,
    strict_mode: bool,
    feature_overrides: dict[str, bool] | None = None,
    sensitivities: list[float] | None = None,
    total_budget: float = 32.0,
    vector_dim: int = 128,
    total_bits_per_value: int = 4,
    outlier_ratio: float = 0.0,
) -> Wave1PolicyProfile:
    decision = resolve_policy(
        requested_family,
        donor_features_enabled=donor_features_enabled,
        strict_mode=strict_mode,
        feature_overrides=feature_overrides,
    )
    ratequant = None
    if decision.selected_family is KvPolicyFamily.ratequant:
        ratequant = ratequant_optimal_bit_allocation(
            sensitivities or [1.0, 1.0, 1.0, 1.0],
            total_budget=total_budget,
        )
    delta_kv = None
    if decision.selected_family is KvPolicyFamily.deltak:
        delta_kv = delta_kv_profile()
    wobble = None
    if decision.selected_family is KvPolicyFamily.wobble:
        wobble = wobble_variance_bit_allocation(
            sensitivities or [1.0, 1.0, 1.0, 1.0],
            total_budget=total_budget,
        )
    qaq = None
    if decision.selected_family is KvPolicyFamily.qaq:
        qaq = qaq_quality_profile(sensitivities or [1.0, 1.0, 1.0, 1.0])
    nautilus = None
    if decision.selected_family is KvPolicyFamily.nautilus:
        nautilus = nautilus_geometric_profile(outlier_ratio=outlier_ratio)
    turbo = None
    if decision.selected_family in {KvPolicyFamily.ratequant, KvPolicyFamily.deltak}:
        turbo = turboquant_profile(
            vector_dim=vector_dim,
            total_bits_per_value=total_bits_per_value,
        )
    guardrail = norm_separated_guardrail(
        requested_bits=total_bits_per_value,
        outlier_ratio=outlier_ratio,
    )
    rkv = None
    if decision.selected_family is KvPolicyFamily.rkv:
        rkv = rkv_redundancy_profile()
    return Wave1PolicyProfile(
        decision=decision,
        ratequant=ratequant,
        delta_kv=delta_kv,
        wobble=wobble,
        qaq=qaq,
        nautilus=nautilus,
        turboquant=turbo,
        guardrail=guardrail,
        rkv=rkv,
    )
