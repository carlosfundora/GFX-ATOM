from dataclasses import dataclass
from enum import Enum


class KvCodec(str, Enum):
    auto = "auto"
    bf16 = "bf16"
    fp8_e4m3 = "fp8_e4m3"
    fp8_e5m2 = "fp8_e5m2"
    int8 = "int8"
    tq1 = "tq1"  # TurboQuant 1-bit (experimental)
    tq2 = "tq2"  # TurboQuant 2-bit (production)
    tq3 = "tq3"  # TurboQuant 3-bit (latency-sensitive)
    tq4 = "tq4"  # TurboQuant 4-bit (accuracy-critical)
    tq8 = "tq8"  # TurboQuant 8-bit (reference)
    rq3_planar = "rq3_planar"
    rq4_planar = "rq4_planar"
    rq3_iso = "rq3_iso"
    rq4_iso = "rq4_iso"


class KvPolicyMode(str, Enum):
    static = "static"
    adaptive = "adaptive"
    learned = "learned"
    fallback = "fallback"


class UniversalKvStage(str, Enum):
    hot_rotor = "hot_rotor"
    warm_rotor_polar = "warm_rotor_polar"
    cold_turbo_residual = "cold_turbo_residual"


@dataclass(frozen=True)
class KvQuantPolicy:
    model_id: str
    codec: KvCodec
    mode: KvPolicyMode
    layer_id: int | None = None
    stage_id: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class UniversalKvBlockHeaderV1:
    block_size: int
    bit_width: int
    rotor_id: int
    codec: KvCodec
    stage: UniversalKvStage
    scale: float
    flags: int = 0
    origin_model_tag: int = 0

    FLAG_PINNED = 0b0000_0001
    FLAG_TURBO_RESIDUAL = 0b0000_0010

    def has_flag(self, flag: int) -> bool:
        return (self.flags & flag) != 0

    def should_apply_turbo_residual(self) -> bool:
        return self.stage == UniversalKvStage.cold_turbo_residual or self.has_flag(
            self.FLAG_TURBO_RESIDUAL
        )


@dataclass(frozen=True)
class UniversalKvPlacementPolicy:
    hot_importance_threshold: float = 0.80
    warm_importance_threshold: float = 0.35
    warm_age_steps: int = 128
    cold_age_steps: int = 512
    gpu_high_watermark: float = 0.90

    def select_stage(
        self, importance: float, age_steps: int, gpu_utilization_pct: float
    ) -> UniversalKvStage:
        importance = min(1.0, max(0.0, float(importance)))
        gpu_utilization_pct = min(1.0, max(0.0, float(gpu_utilization_pct)))

        if (
            importance >= self.hot_importance_threshold
            and age_steps <= self.warm_age_steps
            and gpu_utilization_pct <= self.gpu_high_watermark
        ):
            return UniversalKvStage.hot_rotor

        if importance >= self.warm_importance_threshold and age_steps <= self.cold_age_steps:
            return UniversalKvStage.warm_rotor_polar

        return UniversalKvStage.cold_turbo_residual


def normalize_codec_alias(alias: str) -> KvCodec:
    value = alias.lower()
    if value in {"bf16", "bfloat16"}:
        return KvCodec.bf16
    if value in {"fp8_e4m3", "atom_fp8"}:
        return KvCodec.fp8_e4m3
    if value == "fp8_e5m2":
        return KvCodec.fp8_e5m2
    if value == "int8":
        return KvCodec.int8
    if value in {"turbo_1bit", "tq1"}:
        return KvCodec.tq1
    if value in {"turbo_2bit", "tq2"}:
        return KvCodec.tq2
    if value in {"turbo_3bit", "tq3"}:
        return KvCodec.tq3
    if value in {"turbo_4bit", "tq4"}:
        return KvCodec.tq4
    if value in {"turbo_8bit", "tq8"}:
        return KvCodec.tq8
    if value in {"rq3", "rq3_planar"}:
        return KvCodec.rq3_planar
    if value in {"rq4", "rq4_planar"}:
        return KvCodec.rq4_planar
    if value == "rq3_iso":
        return KvCodec.rq3_iso
    if value == "rq4_iso":
        return KvCodec.rq4_iso
    if value in KvCodec.__members__:
        return KvCodec[value]
    raise ValueError(f"unsupported kv codec alias: {alias}")
