use std::fs;
use std::path::Path;
use std::process::Command;

use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlamaCppCapabilitySurface {
    pub source_root: String,
    pub source_revision: Option<String>,
    pub source_files: Vec<String>,
    pub architectures: Vec<String>,
    pub quantization_labels: Vec<String>,
    pub loader_kv_keys: Vec<String>,
}

impl LlamaCppCapabilitySurface {
    pub fn supports_architecture(&self, architecture: &str) -> bool {
        let normalized = normalize_hf_architecture_name(architecture);
        self.architectures
            .iter()
            .map(|candidate| normalize_symbol(candidate))
            .any(|candidate| candidate == normalized || candidate.starts_with(&normalized) || normalized.starts_with(&candidate))
    }

    pub fn supports_quantization(&self, quantization: &str) -> bool {
        let normalized = normalize_symbol(quantization);
        self.quantization_labels.iter().map(|candidate| normalize_symbol(candidate)).any(|candidate| {
            candidate == normalized || candidate.starts_with(&normalized) || normalized.starts_with(&candidate)
        })
    }
}

pub fn normalize_symbol(value: &str) -> String {
    value
        .trim()
        .to_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

pub fn normalize_hf_architecture_name(value: &str) -> String {
    let lowered = value.trim().to_lowercase();
    let stripped = lowered
        .replace("forcausallm", "")
        .replace("forconditionalgeneration", "")
        .replace("forsequenceclassification", "")
        .replace("fortokenclassification", "");
    normalize_symbol(&stripped)
}

fn capture_list(pattern: &Regex, text: &str) -> Vec<String> {
    let mut items = Vec::new();
    for captures in pattern.captures_iter(text) {
        if let Some(value) = captures.get(1) {
            items.push(value.as_str().trim().to_string());
        }
    }
    items.sort();
    items.dedup();
    items
}

fn detect_git_revision(source_root: &Path) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(source_root)
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let revision = String::from_utf8(output.stdout).ok()?;
    let trimmed = revision.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

pub fn parse_llama_cpp_surface(
    source_root: impl AsRef<Path>,
    arch_source: &str,
    loader_source: &str,
) -> LlamaCppCapabilitySurface {
    let source_root_ref = source_root.as_ref();
    let arch_pattern = Regex::new(r#"\{\s*LLM_ARCH_[A-Z0-9_]+\s*,\s*"([^"]+)""#)
        .expect("valid architecture regex");
    let kv_pattern = Regex::new(r#"\{\s*LLM_KV_[A-Z0-9_]+\s*,\s*"([^"]+)""#)
        .expect("valid kv regex");
    let ftype_pattern = Regex::new(r#"case\s+LLAMA_FTYPE_[A-Z0-9_]+\s*:\s*return\s+"([^"]+)""#)
        .expect("valid ftype regex");

    LlamaCppCapabilitySurface {
        source_root: source_root_ref.display().to_string(),
        source_revision: detect_git_revision(source_root_ref),
        source_files: vec![
            "src/llama-arch.cpp".to_string(),
            "src/llama-model-loader.cpp".to_string(),
        ],
        architectures: capture_list(&arch_pattern, arch_source),
        quantization_labels: capture_list(&ftype_pattern, loader_source),
        loader_kv_keys: {
            let mut kv_keys = capture_list(&kv_pattern, arch_source);
            kv_keys.extend(capture_list(&kv_pattern, loader_source));
            kv_keys.sort();
            kv_keys.dedup();
            kv_keys
        },
    }
}

pub fn load_llama_cpp_surface(source_root: &Path) -> std::io::Result<LlamaCppCapabilitySurface> {
    let arch_source = fs::read_to_string(source_root.join("src/llama-arch.cpp"))?;
    let loader_source = fs::read_to_string(source_root.join("src/llama-model-loader.cpp"))?;
    Ok(parse_llama_cpp_surface(
        source_root,
        &arch_source,
        &loader_source,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        normalize_hf_architecture_name, parse_llama_cpp_surface, LlamaCppCapabilitySurface,
    };

    #[test]
    fn parses_surface_from_source_snippets() {
        let arch_source = r#"
            static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
                { LLM_ARCH_QWEN, "qwen" },
                { LLM_ARCH_QWEN2, "qwen2" },
            };
        "#;
        let loader_source = r#"
            case LLAMA_FTYPE_MOSTLY_Q2_K: return "Q2_K - Medium";
            case LLAMA_FTYPE_MOSTLY_Q4_K_M: return "Q4_K - Medium";
            { LLM_KV_GENERAL_NAME, "general.name" },
            { LLM_KV_ATTENTION_HEAD_COUNT, "%s.attention.head_count" },
        "#;
        let surface = parse_llama_cpp_surface("/tmp/llama.cpp", arch_source, loader_source);
        assert_eq!(surface.source_root, "/tmp/llama.cpp");
        assert!(surface.supports_architecture("Qwen2ForCausalLM"));
        assert!(surface.supports_quantization("q2_k"));
        assert!(surface.loader_kv_keys.iter().any(|key| key == "general.name"));
    }

    #[test]
    fn normalizes_hf_architectures() {
        assert_eq!(normalize_hf_architecture_name("Qwen2ForCausalLM"), "qwen2");
        assert_eq!(normalize_hf_architecture_name("OpenCoderForCausalLM"), "opencoder");
    }

    #[test]
    fn surface_serializes() {
        let surface = LlamaCppCapabilitySurface {
            source_root: "root".to_string(),
            source_revision: Some("deadbeef".to_string()),
            source_files: vec!["src/llama-arch.cpp".to_string()],
            architectures: vec!["qwen2".to_string()],
            quantization_labels: vec!["Q2_K - Medium".to_string()],
            loader_kv_keys: vec!["general.name".to_string()],
        };
        let raw = serde_json::to_string(&surface).expect("serializes");
        let parsed: LlamaCppCapabilitySurface = serde_json::from_str(&raw).expect("parses");
        assert_eq!(parsed.architectures, vec!["qwen2"]);
    }
}
