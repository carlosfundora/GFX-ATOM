use rs_kv_quant_contracts::{normalize_codec_alias, KvCodec, KvCodecError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodecAdapterDescriptor {
    pub codec: KvCodec,
    pub family: String,
    pub backend: String,
    pub supported: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodecBackendPlan {
    pub codec: KvCodec,
    pub family: String,
    pub preferred_backend: String,
    pub fallback_backend: String,
    pub ultimate_fallback: String,
    pub supported: bool,
    pub bit_width: Option<u8>,
    pub is_experimental: bool,
}

impl CodecBackendPlan {
    pub fn backend_chain(&self) -> Vec<String> {
        let mut chain = Vec::new();
        for backend in [
            self.preferred_backend.as_str(),
            self.fallback_backend.as_str(),
            self.ultimate_fallback.as_str(),
        ] {
            let backend = backend.to_string();
            if !chain.contains(&backend) {
                chain.push(backend);
            }
        }
        chain
    }
}

pub trait KvCodecAdapter {
    fn descriptor(&self) -> CodecAdapterDescriptor;
}

#[derive(Debug, Clone)]
pub struct BaselineCodecAdapter {
    codec: KvCodec,
    family: String,
}

impl BaselineCodecAdapter {
    pub fn new(codec: KvCodec, family: impl Into<String>) -> Self {
        Self {
            codec,
            family: family.into(),
        }
    }
}

impl KvCodecAdapter for BaselineCodecAdapter {
    fn descriptor(&self) -> CodecAdapterDescriptor {
        CodecAdapterDescriptor {
            codec: self.codec.clone(),
            family: self.family.clone(),
            backend: "baseline".into(),
            supported: true,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CodecAdapterRegistry {
    adapters: BTreeMap<KvCodec, BaselineCodecAdapter>,
}

impl CodecAdapterRegistry {
    pub fn baseline() -> Self {
        let mut adapters = BTreeMap::new();
        for (alias, family) in [
            ("tq1", "turbo"),
            ("tq8", "turbo"),
            ("tq4", "turbo"),
            ("tq3", "turbo"),
            ("tq2", "turbo"),
            ("rq3_planar", "rotor_planar"),
            ("rq4_planar", "rotor_planar"),
            ("rq3_iso", "rotor_iso"),
            ("rq4_iso", "rotor_iso"),
            ("fp8_e4m3", "fp8"),
        ] {
            if let Ok(codec) = normalize_codec_alias(alias) {
                adapters.insert(codec.clone(), BaselineCodecAdapter::new(codec, family));
            }
        }
        Self { adapters }
    }

    pub fn descriptor_for(&self, codec: &KvCodec) -> Option<CodecAdapterDescriptor> {
        self.adapters.get(codec).map(|adapter| adapter.descriptor())
    }

    pub fn supports(&self, codec: &KvCodec) -> bool {
        self.adapters.contains_key(codec)
    }

    pub fn backend_plan_for(&self, codec: &KvCodec) -> Option<CodecBackendPlan> {
        let descriptor = self.descriptor_for(codec)?;
        match descriptor.family.as_str() {
            "turbo" => self.turboquant_factory(codec),
            "rotor_planar" | "rotor_iso" => self.rotorquant_factory(codec),
            "fp8" => self.fp8_factory(codec),
            "int8" => self.int8_factory(codec),
            _ => None,
        }
    }

    pub fn all_descriptors(&self) -> Vec<CodecAdapterDescriptor> {
        self.adapters.values().map(|adapter| adapter.descriptor()).collect()
    }

    pub fn turboquant_factory(&self, codec: &KvCodec) -> Option<CodecBackendPlan> {
        let descriptor = self.descriptor_for(codec)?;
        if descriptor.family != "turbo" {
            return None;
        }

        Some(CodecBackendPlan {
            codec: codec.clone(),
            family: descriptor.family,
            preferred_backend: "turboquant".into(),
            fallback_backend: "triton".into(),
            ultimate_fallback: "fp16".into(),
            supported: descriptor.supported,
            bit_width: codec.bit_width(),
            is_experimental: matches!(codec, KvCodec::Tq1),
        })
    }

    pub fn rotorquant_factory(&self, codec: &KvCodec) -> Option<CodecBackendPlan> {
        let descriptor = self.descriptor_for(codec)?;
        if !matches!(descriptor.family.as_str(), "rotor_planar" | "rotor_iso") {
            return None;
        }

        Some(CodecBackendPlan {
            codec: codec.clone(),
            family: descriptor.family,
            preferred_backend: "rotorquant".into(),
            fallback_backend: "triton".into(),
            ultimate_fallback: "fp16".into(),
            supported: descriptor.supported,
            bit_width: codec.bit_width(),
            is_experimental: false,
        })
    }

    pub fn fp8_factory(&self, codec: &KvCodec) -> Option<CodecBackendPlan> {
        let descriptor = self.descriptor_for(codec)?;
        if descriptor.family != "fp8" {
            return None;
        }

        Some(CodecBackendPlan {
            codec: codec.clone(),
            family: descriptor.family,
            preferred_backend: "fp8".into(),
            fallback_backend: "triton".into(),
            ultimate_fallback: "fp16".into(),
            supported: descriptor.supported,
            bit_width: codec.bit_width(),
            is_experimental: false,
        })
    }

    pub fn int8_factory(&self, codec: &KvCodec) -> Option<CodecBackendPlan> {
        let descriptor = self.descriptor_for(codec)?;
        if descriptor.family != "int8" {
            return None;
        }

        Some(CodecBackendPlan {
            codec: codec.clone(),
            family: descriptor.family,
            preferred_backend: "int8".into(),
            fallback_backend: "triton".into(),
            ultimate_fallback: "fp16".into(),
            supported: descriptor.supported,
            bit_width: codec.bit_width(),
            is_experimental: false,
        })
    }
}

pub fn normalize_adapter_alias(alias: &str) -> Result<KvCodec, KvCodecError> {
    normalize_codec_alias(alias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_registry_exposes_expected_codecs() {
        let registry = CodecAdapterRegistry::baseline();
        assert!(registry.supports(&KvCodec::Tq4));
        assert!(registry.supports(&KvCodec::Rq3Planar));
        assert!(registry.supports(&KvCodec::Rq4Iso));
        assert!(registry.supports(&KvCodec::Fp8E4M3));
    }

    #[test]
    fn descriptor_reports_family() {
        let registry = CodecAdapterRegistry::baseline();
        let desc = registry.descriptor_for(&KvCodec::Tq3).unwrap();
        assert_eq!(desc.family, "turbo");
        assert!(desc.supported);
        let rotor = registry.descriptor_for(&KvCodec::Rq4Iso).unwrap();
        assert_eq!(rotor.family, "rotor_iso");
    }

    #[test]
    fn turboquant_factory_builds_backend_chain() {
        let registry = CodecAdapterRegistry::baseline();
        let plan = registry.turboquant_factory(&KvCodec::Tq2).unwrap();
        assert_eq!(plan.bit_width, Some(2));
        assert_eq!(plan.backend_chain(), vec!["turboquant", "triton", "fp16"]);
    }

    #[test]
    fn rotorquant_factory_builds_backend_chain() {
        let registry = CodecAdapterRegistry::baseline();
        let plan = registry.rotorquant_factory(&KvCodec::Rq3Iso).unwrap();
        assert_eq!(plan.bit_width, Some(3));
        assert_eq!(plan.backend_chain(), vec!["rotorquant", "triton", "fp16"]);
    }
}
