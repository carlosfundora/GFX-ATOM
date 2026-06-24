from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from kv_policy_arbiter import (  # noqa: E402
    KvPolicyFamily,
    build_wave1_policy_profile,
    resolve_policy,
)


def test_ratequant_requires_global_and_feature_flags():
    decision = resolve_policy(
        "ratequant",
        donor_features_enabled=False,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_RATEQUANT": True},
    )
    assert decision.selected_family == KvPolicyFamily.baseline
    assert decision.accepted is False
    assert decision.rejection_reason == "donor_features_disabled"


def test_ratequant_accepts_when_enabled():
    decision = resolve_policy(
        "ratequant",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_RATEQUANT": True},
    )
    assert decision.selected_family == KvPolicyFamily.ratequant
    assert decision.accepted is True
    assert decision.rejection_reason is None


def test_deltak_accepts_when_enabled():
    decision = resolve_policy(
        "deltak",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_DELTAK": True},
    )
    assert decision.selected_family == KvPolicyFamily.deltak
    assert decision.accepted is True
    assert decision.rejection_reason is None


def test_wobble_accepts_when_enabled():
    decision = resolve_policy(
        "wobble",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_WOBBLE": True},
    )
    assert decision.selected_family == KvPolicyFamily.wobble
    assert decision.accepted is True
    assert decision.rejection_reason is None


def test_qaq_accepts_when_enabled():
    decision = resolve_policy(
        "qaq",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_QAQ": True},
    )
    assert decision.selected_family == KvPolicyFamily.qaq
    assert decision.accepted is True
    assert decision.rejection_reason is None


def test_nautilus_accepts_when_enabled():
    decision = resolve_policy(
        "nautilus",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_NAUTILUS": True},
    )
    assert decision.selected_family == KvPolicyFamily.nautilus
    assert decision.accepted is True
    assert decision.rejection_reason is None


def test_unknown_family_fails_closed():
    decision = resolve_policy(
        "not-a-policy",
        donor_features_enabled=True,
        strict_mode=True,
    )
    assert decision.selected_family == KvPolicyFamily.baseline
    assert decision.accepted is False
    assert decision.rejection_reason is not None


def test_rkv_requires_feature_flag_and_global_flag():
    decision = resolve_policy(
        "rkv",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_RKV": False},
    )
    assert decision.selected_family == KvPolicyFamily.baseline
    assert decision.accepted is False
    assert decision.rejection_reason == "feature_flag_disabled:GFXATOM_KV_RKV"


def test_wave1_profile_includes_ratequant_and_guardrail():
    profile = build_wave1_policy_profile(
        "ratequant",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_RATEQUANT": True},
        sensitivities=[1.0, 8.0],
        total_budget=8.0,
        total_bits_per_value=4,
        outlier_ratio=0.2,
    )
    assert profile.decision.selected_family == KvPolicyFamily.ratequant
    assert profile.ratequant is not None
    assert profile.turboquant is not None
    assert profile.guardrail is not None
    assert profile.guardrail.accepted is False


def test_wave1_profile_includes_deltak_profile():
    profile = build_wave1_policy_profile(
        "deltak",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_DELTAK": True},
        total_bits_per_value=4,
    )
    assert profile.decision.selected_family == KvPolicyFamily.deltak
    assert profile.delta_kv is not None
    assert profile.delta_kv.enabled is True
    assert profile.delta_kv.stores_key_deltas is True
    assert profile.turboquant is not None


def test_wave1_profile_includes_wobble_profile():
    profile = build_wave1_policy_profile(
        "wobble",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_WOBBLE": True},
        sensitivities=[1.0, 16.0],
        total_budget=6.0,
    )
    assert profile.decision.selected_family == KvPolicyFamily.wobble
    assert profile.wobble is not None
    assert profile.wobble.variance_allocations[1] > profile.wobble.variance_allocations[0]


def test_wave1_profile_includes_qaq_profile():
    profile = build_wave1_policy_profile(
        "qaq",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_QAQ": True},
        sensitivities=[0.96, 0.93],
    )
    assert profile.decision.selected_family == KvPolicyFamily.qaq
    assert profile.qaq is not None
    assert profile.qaq.accepted is True
    assert profile.qaq.recommended_bits == 4


def test_wave1_profile_includes_nautilus_profile():
    profile = build_wave1_policy_profile(
        "nautilus",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_NAUTILUS": True},
        outlier_ratio=0.2,
    )
    assert profile.decision.selected_family == KvPolicyFamily.nautilus
    assert profile.nautilus is not None
    assert profile.nautilus.accepted is True
    assert profile.nautilus.note == "high_outlier_ratio_guarded_quant"


def test_wave1_profile_includes_rkv_profile():
    profile = build_wave1_policy_profile(
        "rkv",
        donor_features_enabled=True,
        strict_mode=True,
        feature_overrides={"GFXATOM_KV_RKV": True},
    )
    assert profile.decision.selected_family == KvPolicyFamily.rkv
    assert profile.rkv is not None
    assert profile.rkv.buffer_tokens == 128
    assert profile.ratequant is None
    assert profile.turboquant is None
