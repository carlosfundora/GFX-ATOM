from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from kv_codec_adapters import CodecAdapterRegistry  # noqa: E402
from kv_quant_contracts import KvCodec  # noqa: E402
from sglang_autoquant_bridge import (  # noqa: E402
    AutoQuantPolicySnapshot,
    build_autoquant_backend_summary,
)
from sglang_backend_adapter import SGLangTurboQuantAdapter  # noqa: E402


def _sample_policy() -> AutoQuantPolicySnapshot:
    payload = {
        "fingerprint_digest": "deadbeefcafe1234",
        "model_family": "qwen3-4b",
        "n_layers": 3,
        "version": 1,
        "learner": "calibration_sweep",
        "created_at": 1715920000.0,
        "score": 0.91,
        "layer_codecs": {
            "0": {"codec": "tq2", "bit_width": 2, "note": "stable"},
            "1": {"codec": "rq4_iso", "bit_width": 4, "note": "high variance"},
            "2": {"codec": "fp16", "bit_width": 16, "note": "reference"},
        },
        "stage_overrides": {
            "decode": {
                "1": {"codec": "tq3", "bit_width": 3, "note": "decode-friendly"},
            }
        },
    }
    return AutoQuantPolicySnapshot.from_mapping(payload)


def test_autoquant_policy_snapshot_parses_and_summarizes():
    policy = _sample_policy()

    assert policy.fingerprint_digest == "deadbeefcafe1234"
    assert policy.model_family == "qwen3-4b"
    assert policy.n_layers == 3
    assert policy.is_uniform() is False
    assert policy.codec_histogram() == {"tq2:2": 1, "rq4_iso:4": 1, "fp16:16": 1}
    assert policy.stage_overrides["decode"][1].codec_name == "tq3"

    encoded = json.dumps(policy.to_dict(), sort_keys=True)
    round_trip = AutoQuantPolicySnapshot.from_json(encoded)
    assert round_trip.codec_histogram() == policy.codec_histogram()


def test_autoquant_backend_summary_maps_turbo_layers_to_plan():
    policy = _sample_policy()
    registry = CodecAdapterRegistry()
    summary = build_autoquant_backend_summary(policy, registry=registry)

    assert summary.uniform is False
    assert summary.codec_histogram["tq2:2"] == 1
    assert summary.codec_histogram["rq4_iso:4"] == 1
    assert summary.dispatch[0].backend_chain == ("turboquant", "triton", "fp16")
    assert summary.dispatch[1].backend_chain == ("rotorquant", "triton", "fp16")
    assert summary.dispatch[2].backend_chain == ("native",)


def test_registry_backend_plans_expose_fallback_chains():
    registry = CodecAdapterRegistry()
    turbo_plan = registry.backend_plan_for(KvCodec.tq2)
    rotor_plan = registry.backend_plan_for(KvCodec.rq4_iso)

    assert turbo_plan is not None
    assert turbo_plan.bit_width == 2
    assert turbo_plan.backend_chain() == ("turboquant", "triton", "fp16")
    assert rotor_plan is not None
    assert rotor_plan.bit_width == 4
    assert rotor_plan.backend_chain() == ("rotorquant", "triton", "fp16")


def test_sglang_adapter_exposes_autoquant_summary_when_policy_present():
    policy = _sample_policy()
    adapter = SGLangTurboQuantAdapter(
        kv_cache_dtype_flag="tq2",
        dimension=256,
        num_heads=32,
        autoquant_policy=policy,
    )

    config = adapter.get_config_dict()
    assert config["backend_chain"] == ["turboquant", "triton", "fp16"]
    assert config["autoquant_summary"]["fingerprint_digest"] == "deadbeefcafe1234"
    assert config["autoquant_summary"]["dispatch"][0]["backend_chain"] == [
        "turboquant",
        "triton",
        "fp16",
    ]


def test_sglang_adapter_supports_rotorquant_modes():
    adapter = SGLangTurboQuantAdapter(
        kv_cache_dtype_flag="rq4_iso",
        dimension=256,
        num_heads=32,
    )

    config = adapter.get_config_dict()
    assert config["codec"] == "rq4_iso"
    assert config["backend_chain"] == ["rotorquant", "triton", "fp16"]
    assert config["is_turbo_quant"] is True
    assert config["quant_family"] == "rotor_iso"


def test_sglang_adapter_falls_back_when_turboquant_unavailable():
    adapter = SGLangTurboQuantAdapter(
        kv_cache_dtype_flag="tq2",
        dimension=256,
        num_heads=32,
    )

    assert adapter.resolve_backend_chain(
        turboquant_available=False,
        triton_available=True,
    ) == ("triton", "fp16")
    assert adapter.resolve_backend_chain(
        turboquant_available=False,
        triton_available=False,
    ) == ("fp16",)
