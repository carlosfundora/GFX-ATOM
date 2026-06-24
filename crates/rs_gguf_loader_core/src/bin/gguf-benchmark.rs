use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde::Serialize;

use rs_gguf_loader_core::{
    llama_cpp_family::{load_llama_cpp_family_catalog, select_family_for_architecture, LlamaCppFamilyCatalog},
    llama_cpp_surface::{load_llama_cpp_surface, normalize_hf_architecture_name, LlamaCppCapabilitySurface},
    parse_gguf_header_path,
    profile::{
        GgufProfileBackend, GgufProfileSource, NormalizedGgufProfile, ProfileCatalog,
        select_best_profile,
    },
    synthesize_load_plan,
};

#[derive(Debug, Serialize)]
struct AtomLoadReport {
    load_only: bool,
    elapsed_ms: f64,
    gguf_path: String,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    estimated_index_bytes: u64,
    prefetch_bytes: u64,
    io_chunk_bytes: u64,
    use_mmap: bool,
    use_pinned_staging: bool,
}

#[derive(Debug, Serialize)]
struct LlamaLoadReport {
    ok: bool,
    elapsed_ms: f64,
    exit_code: i32,
    output_tail: String,
}

#[derive(Debug, Serialize)]
struct TurboRotorEstimate {
    model_root: String,
    architecture: String,
    seq_len: u64,
    hidden_size: u64,
    num_attention_heads: u64,
    num_key_value_heads: u64,
    num_hidden_layers: u64,
    head_dim: u64,
    kv_feature_dim: u64,
    gpu_bytes: u64,
    ram_bytes: u64,
    total_bytes: u64,
}

#[derive(Debug, Serialize)]
struct BenchmarkRecord {
    model: String,
    atom: AtomLoadReport,
    llama_cpp: LlamaLoadReport,
    turborotor_vram: TurboRotorEstimate,
    profile_catalog: ProfileCatalog,
    selected_family_id: String,
    selected_profile_id: String,
    selected_profile_source: GgufProfileSource,
    selected_profile_backend: GgufProfileBackend,
}

#[derive(Debug, Serialize)]
struct BenchmarkOutput {
    llama_cpp_surface: LlamaCppCapabilitySurface,
    llama_cpp_family_catalog: LlamaCppFamilyCatalog,
    records: Vec<BenchmarkRecord>,
}

#[derive(Debug, serde::Deserialize)]
struct QwenConfig {
    architectures: Vec<String>,
    hidden_size: u64,
    num_attention_heads: u64,
    num_key_value_heads: u64,
    num_hidden_layers: u64,
}

fn discover_qwen_models() -> Vec<PathBuf> {
    let model_dir = Path::new("/home/local/ai/models/registry/Qwen/Qwen2.5-0.5B-Instruct-GGUF");
    [
        model_dir.join("qwen2.5-0.5b-instruct-q2_k.gguf"),
        model_dir.join("qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ]
    .into_iter()
    .filter(|path| path.exists())
    .collect()
}

fn load_atom_report(model_path: &Path) -> Result<AtomLoadReport, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let header = parse_gguf_header_path(model_path)?;
    let plan = synthesize_load_plan(&header, 64);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(AtomLoadReport {
        load_only: true,
        elapsed_ms,
        gguf_path: model_path.display().to_string(),
        version: header.version,
        tensor_count: header.tensor_count,
        metadata_kv_count: header.metadata_kv_count,
        estimated_index_bytes: header.estimated_index_bytes(),
        prefetch_bytes: plan.prefetch_bytes,
        io_chunk_bytes: plan.io_chunk_bytes,
        use_mmap: plan.use_mmap,
        use_pinned_staging: plan.use_pinned_staging,
    })
}

fn load_llama_report(model_path: &Path) -> LlamaLoadReport {
    let llama_cli = Path::new("/home/local/ai/projects/donors/llama.cpp-1-bit-turbo/build/bin/llama-cli");
    if !llama_cli.exists() {
        return LlamaLoadReport {
            ok: false,
            elapsed_ms: 0.0,
            exit_code: 127,
            output_tail: "llama-cli missing".to_string(),
        };
    }

    let started = Instant::now();
    let output = Command::new(llama_cli)
        .arg("--model")
        .arg(model_path)
        .arg("--ctx-size")
        .arg("128")
        .arg("--threads")
        .arg("4")
        .arg("--predict")
        .arg("1")
        .arg("--prompt")
        .arg("hello")
        .arg("--single-turn")
        .arg("--simple-io")
        .arg("--no-display-prompt")
        .arg("--no-warmup")
        .arg("--device")
        .arg("none")
        .output();

    match output {
        Ok(output) => {
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let combined = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let tail = combined
                .lines()
                .rev()
                .take(12)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n");
            LlamaLoadReport {
                ok: output.status.success(),
                elapsed_ms,
                exit_code: output.status.code().unwrap_or(-1),
                output_tail: tail,
            }
        }
        Err(err) => LlamaLoadReport {
            ok: false,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            exit_code: 127,
            output_tail: err.to_string(),
        },
    }
}

fn estimate_turborotor_vram(model_root: &Path) -> Result<TurboRotorEstimate, Box<dyn std::error::Error>> {
    let model_name = model_root
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("model root missing directory name")?;
    let base_model_dir = if let Some(stripped) = model_name.strip_suffix("-GGUF") {
        model_root
            .parent()
            .ok_or("gguf directory missing parent")?
            .join(stripped)
    } else {
        model_root.to_path_buf()
    };
    let config_path = base_model_dir.join("config.json");
    let config: QwenConfig = serde_json::from_str(&fs::read_to_string(&config_path)?)?;
    let architecture = config
        .architectures
        .first()
        .map(|value| normalize_hf_architecture_name(value))
        .unwrap_or_else(|| "unknown".to_string());
    let seq_len = 128_u64;
    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_feature_dim = config.num_key_value_heads * head_dim * 2;
    let total_values = seq_len
        .saturating_mul(config.num_hidden_layers)
        .saturating_mul(kv_feature_dim);
    let packed_bytes = (total_values.saturating_mul(3) + 7) / 8;
    let block_count = (total_values + 15) / 16;
    let metadata_bytes = block_count.saturating_mul(2);
    let total_bytes = packed_bytes + metadata_bytes;
    let gpu_budget_bytes = 64_u64 * 1024 * 1024;
    let gpu_bytes = total_bytes.min(gpu_budget_bytes);
    let ram_bytes = total_bytes.saturating_sub(gpu_bytes);

    Ok(TurboRotorEstimate {
        model_root: model_root.display().to_string(),
        architecture,
        seq_len,
        hidden_size: config.hidden_size,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        num_hidden_layers: config.num_hidden_layers,
        head_dim,
        kv_feature_dim,
        gpu_bytes,
        ram_bytes,
        total_bytes,
    })
}

fn build_profile_catalog(
    model_path: &Path,
    atom: &AtomLoadReport,
    llama: &LlamaLoadReport,
    vram: &TurboRotorEstimate,
    surface: &LlamaCppCapabilitySurface,
) -> ProfileCatalog {
    let model_name = model_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("unknown-model")
        .to_string();
    let quantization = model_name
        .split('-')
        .next_back()
        .unwrap_or("unknown")
        .replace(".gguf", "");
    let llama_available = surface.supports_architecture(&vram.architecture)
        && surface.supports_quantization(&quantization);
    let atom_profile = NormalizedGgufProfile {
        profile_id: format!("atom:{quantization}"),
        model_name: model_name.clone(),
        architecture: vram.architecture.clone(),
        quantization: quantization.clone(),
        source: GgufProfileSource::AtomDefault,
        backend: GgufProfileBackend::RustAtom,
        available: true,
        completeness_score: 0.95,
        measured_load_ms: Some(atom.elapsed_ms),
        estimated_vram_bytes: Some(vram.total_bytes),
        notes: Some("Rust load-only planner".to_string()),
    };
    let llama_profile = NormalizedGgufProfile {
        profile_id: format!("llama:{quantization}"),
        model_name,
        architecture: vram.architecture.clone(),
        quantization,
        source: GgufProfileSource::LlamaCppSnapshot,
        backend: GgufProfileBackend::LlamaCpp,
        available: llama.ok && llama_available,
        completeness_score: if llama.ok && llama_available { 1.0 } else { 0.25 },
        measured_load_ms: Some(llama.elapsed_ms),
        estimated_vram_bytes: Some(vram.total_bytes),
        notes: Some(
            if llama_available {
                "llama.cpp single-turn load"
            } else {
                "llama.cpp surface missing architecture or quantization support"
            }
            .to_string(),
        ),
    };

    ProfileCatalog {
        catalog_id: "qwen-gguf".to_string(),
        generated_from: "rust-gguf-benchmark".to_string(),
        profiles: vec![atom_profile, llama_profile],
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output_path = PathBuf::from("/home/local/ai/build/wip/gfxATOM-Rust/benchmarks/gguf_load_comparison.json");
    let llama_cpp_source_root = Path::new("/home/local/ai/projects/donors/llama.cpp-1-bit-turbo");
    let llama_cpp_surface = load_llama_cpp_surface(llama_cpp_source_root)?;
    let llama_cpp_family_catalog = load_llama_cpp_family_catalog(llama_cpp_source_root)?;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" => {
                output_path = PathBuf::from(
                    args.next()
                        .ok_or("--output requires a path")?
                );
            }
            value if value.starts_with("--") => {
                return Err(format!("unknown flag: {}", value).into());
            }
            _ => {}
        }
    }

    let mut records = Vec::new();
    for model_path in discover_qwen_models() {
        let model_root = model_path
            .parent()
            .ok_or("model path missing parent directory")?;
        let atom = load_atom_report(&model_path)?;
        let llama = load_llama_report(&model_path);
        let vram = estimate_turborotor_vram(model_root)?;
        let profile_catalog = build_profile_catalog(&model_path, &atom, &llama, &vram, &llama_cpp_surface);
        let selected = select_best_profile(&profile_catalog.profiles)
            .cloned()
            .ok_or("profile catalog unexpectedly empty")?;
        let selected_family_id = select_family_for_architecture(&llama_cpp_family_catalog, &vram.architecture)
            .map(|family| family.family_id.clone())
            .unwrap_or_else(|| vram.architecture.clone());
        records.push(BenchmarkRecord {
            model: model_path.display().to_string(),
            atom,
            llama_cpp: llama,
            turborotor_vram: vram,
            profile_catalog,
            selected_family_id,
            selected_profile_id: selected.profile_id,
            selected_profile_source: selected.source,
            selected_profile_backend: selected.backend,
        });
    }

    let payload = BenchmarkOutput {
        llama_cpp_surface,
        llama_cpp_family_catalog,
        records,
    };
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, serde_json::to_string_pretty(&payload)? + "\n")?;
    println!("{}", serde_json::to_string_pretty(&payload)?);
    println!("saved: {}", output_path.display());
    Ok(())
}
