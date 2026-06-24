use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::llama_cpp_surface::{
    load_llama_cpp_surface, normalize_hf_architecture_name, normalize_symbol,
    LlamaCppCapabilitySurface,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlamaCppFamilyProfile {
    pub family_id: String,
    pub architectures: Vec<String>,
    pub attention_traits: Vec<String>,
    pub graph_traits: Vec<String>,
    pub model_roles: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlamaCppFamilyCatalog {
    pub source_root: String,
    pub source_revision: Option<String>,
    pub source_files: Vec<String>,
    pub attention_feature_markers: Vec<String>,
    pub families: Vec<LlamaCppFamilyProfile>,
}

fn family_id_for_architecture(architecture: &str) -> String {
    let normalized = normalize_symbol(architecture);
    match normalized.as_str() {
        value if value.starts_with("qwen") => "qwen".to_string(),
        value if value.starts_with("lfm2") => "lfm2".to_string(),
        value if value.starts_with("gpt_oss") => "openai_moe".to_string(),
        value if value.starts_with("step35") => "step35".to_string(),
        value if value.starts_with("kimi_linear") => "kimi_linear".to_string(),
        value if value.starts_with("maincoder") => "maincoder".to_string(),
        value if value.starts_with("minimax_m2") => "minimax_m2".to_string(),
        value if value.starts_with("mistral") => "mistral".to_string(),
        value if value.starts_with("deepseek") => "deepseek".to_string(),
        value if value.starts_with("gemma") => "gemma".to_string(),
        value if value.starts_with("mamba") => "mamba".to_string(),
        value if value.starts_with("rwkv") => "rwkv".to_string(),
        value if value.starts_with("phi") => "phi".to_string(),
        value if value.starts_with("llama") => "llama".to_string(),
        // Preserve fork-local or newly added architectures verbatim unless we have a
        // deliberate family mapping for them. That keeps additions like jina/bonsai visible
        // instead of collapsing them into an upstream bucket.
        value => value.to_string(),
    }
}

fn attention_feature_markers(model_source: &str, graph_source: &str) -> Vec<String> {
    let mut markers = Vec::new();
    let source_pairs = [
        ("shared_kv_layers", "shared_kv_layers"),
        ("sliding_window", "sliding_window"),
        ("attention.indexer", "attention.indexer"),
        ("key_length_mla", "key_length_mla"),
        ("value_length_mla", "value_length_mla"),
        ("softcapping", "softcapping"),
        ("pooling_type", "pooling_type"),
        ("linear_attention", "linear_attention"),
        ("moe", "moe"),
        ("vision", "vision"),
    ];

    for (needle, label) in source_pairs {
        if model_source.contains(needle) || graph_source.contains(needle) {
            markers.push(label.to_string());
        }
    }

    markers.sort();
    markers.dedup();
    markers
}

fn attention_traits_for_family(family_id: &str, architectures: &[String], source_markers: &[String]) -> Vec<String> {
    let mut traits = Vec::new();

    if architectures.iter().any(|architecture| architecture.contains("moe")) || family_id == "openai_moe" {
        traits.push("moe_routing".to_string());
    }

    if family_id == "qwen" {
        traits.extend([
            "last_token_pooling".to_string(),
            "shared_kv_layers".to_string(),
            "sliding_window".to_string(),
        ]);
    }

    if family_id == "lfm2" {
        traits.extend([
            "hybrid_context".to_string(),
            "moe_routing".to_string(),
        ]);
    }

    if family_id == "step35" {
        traits.extend([
            "stepwise_graph".to_string(),
            "special_cache_path".to_string(),
        ]);
    }

    if family_id == "kimi_linear" {
        traits.extend([
            "linear_attention".to_string(),
            "long_context".to_string(),
        ]);
    }

    if family_id == "maincoder" {
        traits.push("code_generation".to_string());
    }

    if family_id == "minimax_m2" {
        traits.push("reasoning".to_string());
    }

    if family_id == "mistral" {
        traits.extend([
            "sliding_window".to_string(),
            "softcapping".to_string(),
        ]);
    }

    if family_id == "deepseek" {
        traits.extend([
            "yarn_rope".to_string(),
            "special_attention_indexer".to_string(),
        ]);
    }

    if family_id == "gemma" {
        traits.extend([
            "vision_aware".to_string(),
            "embedding_pooling".to_string(),
        ]);
    }

    if source_markers.iter().any(|marker| marker == "attention.indexer") {
        traits.push("indexed_attention".to_string());
    }

    if source_markers.iter().any(|marker| marker == "shared_kv_layers") {
        traits.push("shared_kv_layers".to_string());
    }

    if source_markers.iter().any(|marker| marker == "sliding_window") {
        traits.push("sliding_window".to_string());
    }

    traits.sort();
    traits.dedup();
    traits
}

fn graph_traits_for_family(family_id: &str, architectures: &[String], source_markers: &[String]) -> Vec<String> {
    let mut traits = Vec::new();

    if architectures.iter().any(|architecture| architecture.contains("vl")) || family_id == "gemma" {
        traits.push("multimodal".to_string());
    }

    if architectures.iter().any(|architecture| architecture.contains("embedding")) {
        traits.push("embedding".to_string());
    }

    if architectures.iter().any(|architecture| architecture.contains("ocr")) || family_id == "deepseek" {
        traits.push("ocr_or_document".to_string());
    }

    if architectures.iter().any(|architecture| architecture.contains("moe")) {
        traits.push("mixture_of_experts".to_string());
    }

    if family_id == "step35" {
        traits.push("special_graph_branch".to_string());
    }

    if source_markers.iter().any(|marker| marker == "pooling_type") {
        traits.push("pooling".to_string());
    }

    traits.sort();
    traits.dedup();
    traits
}

fn model_roles_for_family(family_id: &str, architectures: &[String]) -> Vec<String> {
    let mut roles = Vec::new();

    if family_id == "maincoder" {
        roles.push("code".to_string());
    }

    if family_id == "qwen" || family_id == "mistral" || family_id == "llama" {
        roles.push("chat".to_string());
    }

    if family_id == "lfm2" || family_id == "minimax_m2" || architectures.iter().any(|architecture| architecture.contains("thinking")) {
        roles.push("reasoning".to_string());
    }

    if architectures.iter().any(|architecture| architecture.contains("embedding")) {
        roles.push("embedding".to_string());
    }

    if architectures.iter().any(|architecture| architecture.contains("vl")) || architectures.iter().any(|architecture| architecture.contains("ocr")) {
        roles.push("multimodal".to_string());
    }

    if roles.is_empty() {
        roles.push("general".to_string());
    }

    roles.sort();
    roles.dedup();
    roles
}

pub fn build_llama_cpp_family_catalog(
    surface: &LlamaCppCapabilitySurface,
    model_source: &str,
    graph_source: &str,
) -> LlamaCppFamilyCatalog {
    let source_markers = attention_feature_markers(model_source, graph_source);
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for architecture in &surface.architectures {
        let family_id = family_id_for_architecture(architecture);
        grouped.entry(family_id).or_default().push(architecture.clone());
    }

    let families = grouped
        .into_iter()
        .map(|(family_id, mut architectures)| {
            architectures.sort();
            architectures.dedup();
            let attention_traits = attention_traits_for_family(&family_id, &architectures, &source_markers);
            let graph_traits = graph_traits_for_family(&family_id, &architectures, &source_markers);
            let model_roles = model_roles_for_family(&family_id, &architectures);

            LlamaCppFamilyProfile {
                family_id,
                architectures,
                attention_traits,
                graph_traits,
                model_roles,
            }
        })
        .collect::<Vec<_>>();

    LlamaCppFamilyCatalog {
        source_root: surface.source_root.clone(),
        source_revision: surface.source_revision.clone(),
        source_files: {
            let mut files = surface.source_files.clone();
            files.push("src/llama-model.cpp".to_string());
            files.push("src/llama-graph.cpp".to_string());
            files.sort();
            files.dedup();
            files
        },
        attention_feature_markers: source_markers,
        families,
    }
}

pub fn load_llama_cpp_family_catalog(source_root: &Path) -> std::io::Result<LlamaCppFamilyCatalog> {
    let surface = load_llama_cpp_surface(source_root)?;
    let model_source = fs::read_to_string(source_root.join("src/llama-model.cpp")).unwrap_or_default();
    let graph_source = fs::read_to_string(source_root.join("src/llama-graph.cpp")).unwrap_or_default();
    Ok(build_llama_cpp_family_catalog(&surface, &model_source, &graph_source))
}

pub fn select_family_for_architecture<'a>(
    catalog: &'a LlamaCppFamilyCatalog,
    architecture: &str,
) -> Option<&'a LlamaCppFamilyProfile> {
    let normalized = normalize_hf_architecture_name(architecture);
    catalog
        .families
        .iter()
        .find(|family| {
            family
                .architectures
                .iter()
                .any(|candidate| normalize_hf_architecture_name(candidate) == normalized)
        })
}

#[cfg(test)]
mod tests {
    use super::{
        build_llama_cpp_family_catalog, select_family_for_architecture, LlamaCppFamilyCatalog,
    };
    use crate::llama_cpp_surface::LlamaCppCapabilitySurface;

    #[test]
    fn groups_architectures_into_families_with_traits() {
        let surface = LlamaCppCapabilitySurface {
            source_root: "/tmp/llama.cpp".to_string(),
            source_revision: Some("deadbeef".to_string()),
            source_files: vec!["src/llama-arch.cpp".to_string(), "src/llama-model-loader.cpp".to_string()],
            architectures: vec![
                "qwen3".to_string(),
                "qwen35moe".to_string(),
                "lfm2moe".to_string(),
                "kimi-linear".to_string(),
                "gpt-oss".to_string(),
                "step35".to_string(),
                "deepseek2-ocr".to_string(),
                "jina-embeddings-v3".to_string(),
                "bonsai".to_string(),
            ],
            quantization_labels: vec![],
            loader_kv_keys: vec![],
        };
        let catalog = build_llama_cpp_family_catalog(&surface, "shared_kv_layers", "attention.indexer");
        assert!(catalog.families.iter().any(|family| family.family_id == "qwen"));
        assert!(catalog.families.iter().any(|family| family.family_id == "lfm2"));
        assert!(catalog.families.iter().any(|family| family.family_id == "kimi_linear"));
        assert!(catalog.families.iter().any(|family| family.family_id == "openai_moe"));
        assert!(catalog.families.iter().any(|family| family.family_id == "jina_embeddings_v3"));
        assert!(catalog.families.iter().any(|family| family.family_id == "bonsai"));
        let qwen = catalog
            .families
            .iter()
            .find(|family| family.family_id == "qwen")
            .expect("qwen family");
        assert!(qwen.attention_traits.contains(&"sliding_window".to_string()));
        assert!(qwen.attention_traits.contains(&"shared_kv_layers".to_string()));
        let deepseek = catalog
            .families
            .iter()
            .find(|family| family.family_id == "deepseek")
            .expect("deepseek family");
        assert!(deepseek.graph_traits.contains(&"ocr_or_document".to_string()));
        assert!(deepseek.attention_traits.contains(&"indexed_attention".to_string()));
    }

    #[test]
    fn selects_family_for_architecture() {
        let catalog = LlamaCppFamilyCatalog {
            source_root: "root".to_string(),
            source_revision: None,
            source_files: vec![],
            attention_feature_markers: vec![],
            families: vec![super::LlamaCppFamilyProfile {
                family_id: "qwen".to_string(),
                architectures: vec!["qwen2".to_string()],
                attention_traits: vec![],
                graph_traits: vec![],
                model_roles: vec![],
            }],
        };
        let selected = select_family_for_architecture(&catalog, "Qwen2ForCausalLM").expect("family selected");
        assert_eq!(selected.family_id, "qwen");
    }

    #[test]
    fn serializes_catalog() {
        let catalog = LlamaCppFamilyCatalog {
            source_root: "root".to_string(),
            source_revision: Some("deadbeef".to_string()),
            source_files: vec!["src/llama-arch.cpp".to_string()],
            attention_feature_markers: vec!["sliding_window".to_string()],
            families: vec![super::LlamaCppFamilyProfile {
                family_id: "qwen".to_string(),
                architectures: vec!["qwen2".to_string()],
                attention_traits: vec!["sliding_window".to_string()],
                graph_traits: vec!["pooling".to_string()],
                model_roles: vec!["chat".to_string()],
            }],
        };
        let raw = serde_json::to_string(&catalog).expect("serializes");
        let parsed: LlamaCppFamilyCatalog = serde_json::from_str(&raw).expect("parses");
        assert_eq!(parsed.families.len(), 1);
    }
}
