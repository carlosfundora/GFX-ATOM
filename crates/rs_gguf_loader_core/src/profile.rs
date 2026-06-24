use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GgufProfileSource {
    LlamaCppSnapshot,
    SglangSnapshot,
    AtomDefault,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GgufProfileBackend {
    RustAtom,
    LlamaCpp,
    Sglang,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalizedGgufProfile {
    pub profile_id: String,
    pub model_name: String,
    pub architecture: String,
    pub quantization: String,
    pub source: GgufProfileSource,
    pub backend: GgufProfileBackend,
    pub available: bool,
    pub completeness_score: f32,
    pub measured_load_ms: Option<f64>,
    pub estimated_vram_bytes: Option<u64>,
    pub notes: Option<String>,
}

impl NormalizedGgufProfile {
    pub fn score(&self) -> (u8, u64, u32, u8) {
        let availability_rank = if self.available { 0 } else { 1 };
        let measured_rank = self
            .measured_load_ms
            .map(|value| (value.max(0.0) * 1_000.0).round() as u64)
            .unwrap_or(u64::MAX);
        let completeness_rank =
            (1_000.0 - (self.completeness_score.clamp(0.0, 1.0) * 1_000.0)).round() as u32;
        let source_rank = match self.source {
            GgufProfileSource::LlamaCppSnapshot => 0,
            GgufProfileSource::SglangSnapshot => 1,
            GgufProfileSource::AtomDefault => 2,
        };
        (availability_rank, measured_rank, completeness_rank, source_rank)
    }
}

pub fn select_best_profile(
    profiles: &[NormalizedGgufProfile],
) -> Option<&NormalizedGgufProfile> {
    profiles.iter().min_by(|left, right| left.score().cmp(&right.score()))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProfileCatalog {
    pub catalog_id: String,
    pub generated_from: String,
    pub profiles: Vec<NormalizedGgufProfile>,
}

impl ProfileCatalog {
    pub fn best_for_model(&self, model_name: &str) -> Option<&NormalizedGgufProfile> {
        let matching: Vec<&NormalizedGgufProfile> = self
            .profiles
            .iter()
            .filter(|profile| profile.model_name == model_name)
            .collect();
        matching.into_iter().min_by(|left, right| left.score().cmp(&right.score()))
    }
}

pub fn parse_profile_catalog_json(
    raw: &str,
) -> Result<ProfileCatalog, serde_json::Error> {
    serde_json::from_str(raw)
}

#[cfg(test)]
mod tests {
    use super::{
        parse_profile_catalog_json, select_best_profile, GgufProfileBackend, GgufProfileSource,
        NormalizedGgufProfile, ProfileCatalog,
    };

    #[test]
    fn prefers_faster_available_profile() {
        let llama = NormalizedGgufProfile {
            profile_id: "llama:qwen".to_string(),
            model_name: "qwen".to_string(),
            architecture: "qwen2".to_string(),
            quantization: "q2_k".to_string(),
            source: GgufProfileSource::LlamaCppSnapshot,
            backend: GgufProfileBackend::RustAtom,
            available: true,
            completeness_score: 1.0,
            measured_load_ms: Some(10.0),
            estimated_vram_bytes: Some(1),
            notes: None,
        };
        let sglang = NormalizedGgufProfile {
            profile_id: "sglang:qwen".to_string(),
            model_name: "qwen".to_string(),
            architecture: "qwen2".to_string(),
            quantization: "q2_k".to_string(),
            source: GgufProfileSource::SglangSnapshot,
            backend: GgufProfileBackend::Sglang,
            available: true,
            completeness_score: 1.0,
            measured_load_ms: Some(20.0),
            estimated_vram_bytes: Some(1),
            notes: None,
        };
        let profiles = [sglang, llama];
        let selected = select_best_profile(&profiles).expect("selection should exist");
        assert_eq!(selected.profile_id, "llama:qwen");
    }

    #[test]
    fn catalog_filters_by_model() {
        let profile = NormalizedGgufProfile {
            profile_id: "atom:qwen".to_string(),
            model_name: "qwen".to_string(),
            architecture: "qwen2".to_string(),
            quantization: "q4_k_m".to_string(),
            source: GgufProfileSource::AtomDefault,
            backend: GgufProfileBackend::RustAtom,
            available: true,
            completeness_score: 0.8,
            measured_load_ms: None,
            estimated_vram_bytes: None,
            notes: None,
        };
        let catalog = ProfileCatalog {
            catalog_id: "test".to_string(),
            generated_from: "unit-test".to_string(),
            profiles: vec![profile],
        };
        assert_eq!(
            catalog.best_for_model("qwen").map(|row| row.profile_id.as_str()),
            Some("atom:qwen")
        );
    }

    #[test]
    fn catalog_round_trips_from_json() {
        let catalog = ProfileCatalog {
            catalog_id: "roundtrip".to_string(),
            generated_from: "unit-test".to_string(),
            profiles: vec![NormalizedGgufProfile {
                profile_id: "atom:qwen".to_string(),
                model_name: "qwen".to_string(),
                architecture: "gguf".to_string(),
                quantization: "q2_k".to_string(),
                source: GgufProfileSource::AtomDefault,
                backend: GgufProfileBackend::RustAtom,
                available: true,
                completeness_score: 0.9,
                measured_load_ms: Some(1.0),
                estimated_vram_bytes: Some(123),
                notes: Some("test".to_string()),
            }],
        };
        let raw = serde_json::to_string(&catalog).expect("serializes");
        let parsed = parse_profile_catalog_json(&raw).expect("parses");
        assert_eq!(parsed.catalog_id, "roundtrip");
        assert_eq!(parsed.profiles.len(), 1);
    }
}
