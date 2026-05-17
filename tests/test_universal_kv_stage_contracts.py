from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from kv_quant_contracts import (  # noqa: E402
    KvCodec,
    UniversalKvBlockHeaderV1,
    UniversalKvPlacementPolicy,
    UniversalKvStage,
)


def test_stage_policy_prefers_hot_for_recent_important_blocks():
    policy = UniversalKvPlacementPolicy()
    stage = policy.select_stage(importance=0.95, age_steps=12, gpu_utilization_pct=0.50)
    assert stage is UniversalKvStage.hot_rotor


def test_stage_policy_demotes_under_gpu_pressure():
    policy = UniversalKvPlacementPolicy()
    stage = policy.select_stage(importance=0.95, age_steps=12, gpu_utilization_pct=0.98)
    assert stage is UniversalKvStage.warm_rotor_polar
    cold = policy.select_stage(importance=0.05, age_steps=4096, gpu_utilization_pct=0.98)
    assert cold is UniversalKvStage.cold_turbo_residual


def test_universal_block_header_flag_logic():
    header = UniversalKvBlockHeaderV1(
        block_size=16,
        bit_width=3,
        rotor_id=4,
        codec=KvCodec.rq3_planar,
        stage=UniversalKvStage.warm_rotor_polar,
        scale=0.25,
        flags=UniversalKvBlockHeaderV1.FLAG_TURBO_RESIDUAL,
        origin_model_tag=11,
    )
    assert header.has_flag(UniversalKvBlockHeaderV1.FLAG_TURBO_RESIDUAL)
    assert header.should_apply_turbo_residual() is True
