from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from wave1_donor_adapters import (  # noqa: E402
    delta_kv_profile,
    cpu_bench_summary_from_requests,
    norm_separated_guardrail,
    nautilus_geometric_profile,
    qaq_quality_profile,
    ratequant_optimal_bit_allocation,
    rkv_redundancy_profile,
    wobble_variance_bit_allocation,
    turboquant_profile,
)


def test_ratequant_allocation_biases_high_sensitivity():
    profile = ratequant_optimal_bit_allocation([1.0, 16.0], total_budget=6.0)
    assert len(profile.bit_allocations) == 2
    assert profile.bit_allocations[1] > profile.bit_allocations[0]


def test_norm_guardrail_rejects_low_bit_high_outlier():
    guard = norm_separated_guardrail(requested_bits=4, outlier_ratio=0.2)
    assert guard.accepted is False
    assert guard.recommended_bits == 8
    assert guard.reason == "low_bit_outlier_risk"


def test_turboquant_profile_split_is_consistent():
    profile = turboquant_profile(vector_dim=128, total_bits_per_value=4)
    assert profile.polar_bits_per_value == 3
    assert profile.qjl_bits_per_value == 1
    assert profile.qjl_projections > 0


def test_delta_kv_profile_marks_key_delta_mode():
    profile = delta_kv_profile()
    assert profile.enabled is True
    assert profile.stores_key_deltas is True
    assert profile.note == "delta_kv_enabled"


def test_wobble_allocation_biases_high_variance():
    profile = wobble_variance_bit_allocation([1.0, 16.0], total_budget=6.0)
    assert len(profile.variance_allocations) == 2
    assert profile.variance_allocations[1] > profile.variance_allocations[0]


def test_qaq_profile_tracks_quality_threshold():
    profile = qaq_quality_profile([0.96, 0.93], target_quality=0.9)
    assert profile.accepted is True
    assert profile.recommended_bits == 4
    assert profile.note == "quality_target_met"

    rejected = qaq_quality_profile([0.4, 0.6], target_quality=0.9)
    assert rejected.accepted is False
    assert rejected.recommended_bits == 8
    assert rejected.note == "quality_target_missed"


def test_nautilus_profile_uses_golden_ratio_ladder():
    profile = nautilus_geometric_profile(outlier_ratio=0.2)
    assert profile.accepted is True
    assert profile.note == "high_outlier_ratio_guarded_quant"
    assert len(profile.geometric_weights) == 4
    assert profile.geometric_weights[-1] > profile.geometric_weights[0]


def test_cpu_bench_summary_aggregates_records():
    summary = cpu_bench_summary_from_requests(
        [
            {"e2e_ms": 10.0, "tokens_per_second": 120.0},
            {"e2e_ms": 30.0, "tokens_per_second": 80.0},
        ]
    )
    assert summary.num_requests == 2
    assert summary.avg_e2e_ms == 20.0
    assert summary.p95_e2e_ms == 10.0
    assert summary.avg_tokens_per_second == 100.0


def test_rkv_profile_captures_decode_time_compression_parameters():
    profile = rkv_redundancy_profile(
        buffer_tokens=128,
        observation_tokens=32,
        lambda_weight=0.6,
        redundancy_window=4,
        top_k_tokens=64,
    )
    assert profile.buffer_tokens == 128
    assert profile.observation_tokens == 32
    assert profile.lambda_weight == 0.6
    assert profile.redundancy_window == 4
    assert profile.top_k_tokens == 64
