from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

from kv_codec_adapters import CodecAdapterRegistry
from kv_quant_contracts import KvCodec, normalize_codec_alias


@dataclass(frozen=True)
class AutoQuantLayerPolicy:
    codec_name: str
    bit_width: int
    note: str | None = None

    @property
    def codec(self) -> KvCodec:
        return normalize_codec_alias(self.codec_name)

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": self.codec_name,
            "bit_width": self.bit_width,
            "note": self.note,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "AutoQuantLayerPolicy":
        return cls(
            codec_name=str(payload.get("codec", "auto")),
            bit_width=int(payload.get("bit_width", 0)),
            note=str(payload.get("note")) if payload.get("note") is not None else None,
        )


@dataclass(frozen=True)
class AutoQuantPolicySnapshot:
    fingerprint_digest: str
    model_family: str
    n_layers: int
    version: int
    learner: str
    created_at: float
    score: float
    layer_codecs: dict[int, AutoQuantLayerPolicy]
    stage_overrides: dict[str, dict[int, AutoQuantLayerPolicy]]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "AutoQuantPolicySnapshot":
        layer_raw = payload.get("layer_codecs", {})
        stage_raw = payload.get("stage_overrides", {})

        layer_codecs = {
            int(layer): AutoQuantLayerPolicy.from_mapping(config)
            for layer, config in dict(layer_raw).items()
        }
        stage_overrides: dict[str, dict[int, AutoQuantLayerPolicy]] = {}
        for stage, entries in dict(stage_raw).items():
            stage_overrides[str(stage)] = {
                int(layer): AutoQuantLayerPolicy.from_mapping(config)
                for layer, config in dict(entries).items()
            }

        return cls(
            fingerprint_digest=str(payload.get("fingerprint_digest", "")),
            model_family=str(payload.get("model_family", "")),
            n_layers=int(payload.get("n_layers", len(layer_codecs))),
            version=int(payload.get("version", 1)),
            learner=str(payload.get("learner", "unknown")),
            created_at=float(payload.get("created_at", 0.0)),
            score=float(payload.get("score", 0.0)),
            layer_codecs=layer_codecs,
            stage_overrides=stage_overrides,
        )

    @classmethod
    def from_json(cls, raw: str) -> "AutoQuantPolicySnapshot":
        payload = json.loads(raw)
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "fingerprint_digest": self.fingerprint_digest,
            "model_family": self.model_family,
            "n_layers": self.n_layers,
            "version": self.version,
            "learner": self.learner,
            "created_at": self.created_at,
            "score": self.score,
            "layer_codecs": {
                str(layer): policy.to_dict()
                for layer, policy in sorted(self.layer_codecs.items())
            },
            "stage_overrides": {
                stage: {
                    str(layer): policy.to_dict()
                    for layer, policy in sorted(entries.items())
                }
                for stage, entries in sorted(self.stage_overrides.items())
            },
        }

    def codec_histogram(self) -> dict[str, int]:
        histogram: dict[str, int] = {}
        for layer in self.layer_codecs.values():
            key = f"{layer.codec_name}:{layer.bit_width}"
            histogram[key] = histogram.get(key, 0) + 1
        return histogram

    def is_uniform(self) -> bool:
        if not self.layer_codecs:
            return True
        first = next(iter(self.layer_codecs.values()))
        return all(
            item.codec_name == first.codec_name and item.bit_width == first.bit_width
            for item in self.layer_codecs.values()
        )


@dataclass(frozen=True)
class AutoQuantDispatchEntry:
    layer_id: int
    codec: str
    bit_width: int
    backend_chain: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "codec": self.codec,
            "bit_width": self.bit_width,
            "backend_chain": list(self.backend_chain),
        }


@dataclass(frozen=True)
class AutoQuantBackendSummary:
    fingerprint_digest: str
    model_family: str
    uniform: bool
    codec_histogram: dict[str, int]
    dispatch: list[AutoQuantDispatchEntry]

    def to_dict(self) -> dict[str, object]:
        return {
            "fingerprint_digest": self.fingerprint_digest,
            "model_family": self.model_family,
            "uniform": self.uniform,
            "codec_histogram": dict(self.codec_histogram),
            "dispatch": [entry.to_dict() for entry in self.dispatch],
        }


def _backend_chain_for_layer(policy: AutoQuantLayerPolicy, registry: CodecAdapterRegistry) -> tuple[str, ...]:
    try:
        resolved_codec = normalize_codec_alias(policy.codec_name)
    except ValueError:
        return ("native",)
    plan = registry.backend_plan_for(resolved_codec)
    if plan is None:
        return ("native",)
    return plan.backend_chain()


def build_autoquant_backend_summary(
    policy: AutoQuantPolicySnapshot,
    registry: CodecAdapterRegistry | None = None,
) -> AutoQuantBackendSummary:
    resolved_registry = registry or CodecAdapterRegistry()
    dispatch: list[AutoQuantDispatchEntry] = []
    for layer_id, layer_policy in sorted(policy.layer_codecs.items()):
        dispatch.append(
            AutoQuantDispatchEntry(
                layer_id=layer_id,
                codec=layer_policy.codec_name,
                bit_width=layer_policy.bit_width,
                backend_chain=_backend_chain_for_layer(layer_policy, resolved_registry),
            )
        )

    return AutoQuantBackendSummary(
        fingerprint_digest=policy.fingerprint_digest,
        model_family=policy.model_family,
        uniform=policy.is_uniform(),
        codec_histogram=policy.codec_histogram(),
        dispatch=dispatch,
    )
