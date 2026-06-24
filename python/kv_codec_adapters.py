from dataclasses import dataclass

from typing import Optional

from kv_quant_contracts import KvCodec, normalize_codec_alias


@dataclass(frozen=True)
class CodecAdapterDescriptor:
    codec: KvCodec
    family: str
    backend: str
    supported: bool = True


@dataclass(frozen=True)
class CodecBackendPlan:
    codec: KvCodec
    family: str
    preferred_backend: str
    fallback_backend: str
    ultimate_fallback: str
    supported: bool = True
    bit_width: Optional[int] = None
    is_experimental: bool = False

    def backend_chain(self) -> tuple[str, ...]:
        chain: list[str] = []
        for backend in (
            self.preferred_backend,
            self.fallback_backend,
            self.ultimate_fallback,
        ):
            if backend and backend not in chain:
                chain.append(backend)
        return tuple(chain)


class CodecAdapterRegistry:
    _baseline = {
        KvCodec.tq1: CodecAdapterDescriptor(KvCodec.tq1, "turbo", "baseline"),
        KvCodec.tq2: CodecAdapterDescriptor(KvCodec.tq2, "turbo", "baseline"),
        KvCodec.tq3: CodecAdapterDescriptor(KvCodec.tq3, "turbo", "baseline"),
        KvCodec.tq4: CodecAdapterDescriptor(KvCodec.tq4, "turbo", "baseline"),
        KvCodec.tq8: CodecAdapterDescriptor(KvCodec.tq8, "turbo", "baseline"),
        KvCodec.rq3_planar: CodecAdapterDescriptor(KvCodec.rq3_planar, "rotor_planar", "baseline"),
        KvCodec.rq4_planar: CodecAdapterDescriptor(KvCodec.rq4_planar, "rotor_planar", "baseline"),
        KvCodec.rq3_iso: CodecAdapterDescriptor(KvCodec.rq3_iso, "rotor_iso", "baseline"),
        KvCodec.rq4_iso: CodecAdapterDescriptor(KvCodec.rq4_iso, "rotor_iso", "baseline"),
        KvCodec.fp8_e4m3: CodecAdapterDescriptor(KvCodec.fp8_e4m3, "fp8", "baseline"),
        KvCodec.fp8_e5m2: CodecAdapterDescriptor(KvCodec.fp8_e5m2, "fp8", "baseline"),
        KvCodec.int8: CodecAdapterDescriptor(KvCodec.int8, "int8", "baseline"),
    }

    def descriptor_for(self, codec: KvCodec) -> CodecAdapterDescriptor | None:
        return self._baseline.get(codec)

    def supports(self, codec: KvCodec) -> bool:
        return codec in self._baseline

    def all_descriptors(self) -> list[CodecAdapterDescriptor]:
        return list(self._baseline.values())

    def backend_plan_for(self, codec: KvCodec) -> CodecBackendPlan | None:
        descriptor = self.descriptor_for(codec)
        if descriptor is None:
            return None
        if descriptor.family == "turbo":
            return self.turboquant_factory(codec)
        if descriptor.family.startswith("rotor"):
            return self.rotorquant_factory(codec)
        if descriptor.family == "fp8":
            return self.fp8_factory(codec)
        if descriptor.family == "int8":
            return self.int8_factory(codec)
        return None

    def turboquant_factory(self, codec: KvCodec) -> CodecBackendPlan | None:
        descriptor = self.descriptor_for(codec)
        if descriptor is None or descriptor.family != "turbo":
            return None

        bit_width: Optional[int] = None
        if codec.value.startswith("tq") and codec.value[2:].isdigit():
            bit_width = int(codec.value[2:])

        return CodecBackendPlan(
            codec=codec,
            family=descriptor.family,
            preferred_backend="turboquant",
            fallback_backend="triton",
            ultimate_fallback="fp16",
            supported=descriptor.supported,
            bit_width=bit_width,
            is_experimental=codec is KvCodec.tq1,
        )

    def rotorquant_factory(self, codec: KvCodec) -> CodecBackendPlan | None:
        descriptor = self.descriptor_for(codec)
        if descriptor is None or not descriptor.family.startswith("rotor"):
            return None

        bit_width: Optional[int] = None
        if codec.value.startswith("rq") and codec.value[2:].split("_", 1)[0].isdigit():
            bit_width = int(codec.value[2])

        return CodecBackendPlan(
            codec=codec,
            family=descriptor.family,
            preferred_backend="rotorquant",
            fallback_backend="triton",
            ultimate_fallback="fp16",
            supported=descriptor.supported,
            bit_width=bit_width,
            is_experimental=False,
        )

    def fp8_factory(self, codec: KvCodec) -> CodecBackendPlan | None:
        descriptor = self.descriptor_for(codec)
        if descriptor is None or descriptor.family != "fp8":
            return None
        return CodecBackendPlan(
            codec=codec,
            family=descriptor.family,
            preferred_backend="fp8",
            fallback_backend="triton",
            ultimate_fallback="fp16",
            supported=descriptor.supported,
            bit_width=8,
            is_experimental=False,
        )

    def int8_factory(self, codec: KvCodec) -> CodecBackendPlan | None:
        descriptor = self.descriptor_for(codec)
        if descriptor is None or descriptor.family != "int8":
            return None
        return CodecBackendPlan(
            codec=codec,
            family=descriptor.family,
            preferred_backend="int8",
            fallback_backend="triton",
            ultimate_fallback="fp16",
            supported=descriptor.supported,
            bit_width=8,
            is_experimental=False,
        )


def normalize_adapter_alias(alias: str) -> KvCodec:
    return normalize_codec_alias(alias)
