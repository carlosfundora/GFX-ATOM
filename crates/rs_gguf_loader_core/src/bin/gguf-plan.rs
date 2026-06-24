use std::env;
use std::path::PathBuf;

use rs_gguf_loader_core::{parse_gguf_header_path, synthesize_load_plan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut load_only = false;
    let mut emit_json = false;
    let mut positional: Vec<String> = Vec::new();
    let mut max_prefetch_mb: u64 = 64;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--load-only" => load_only = true,
            "--json" => emit_json = true,
            "--max-prefetch-mb" => {
                max_prefetch_mb = args
                    .next()
                    .as_deref()
                    .ok_or("--max-prefetch-mb requires a value")?
                    .parse::<u64>()?;
            }
            value if value.starts_with("--") => {
                return Err(format!("unknown flag: {}", value).into());
            }
            value => {
                positional.push(value.to_string());
            }
        }
    }

    if positional.is_empty() {
        return Err("usage: gguf-plan [--load-only] [--json] <model.gguf> [max_prefetch_mb]".into());
    }
    if positional.len() > 2 {
        return Err("usage: gguf-plan [--load-only] [--json] <model.gguf> [max_prefetch_mb]".into());
    }

    let gguf_path = PathBuf::from(&positional[0]);
    if let Some(value) = positional.get(1) {
        max_prefetch_mb = value.parse::<u64>()?;
    }

    let header = parse_gguf_header_path(&gguf_path)?;
    let plan = synthesize_load_plan(&header, max_prefetch_mb);

    if emit_json {
        println!(
            "{{\"load_only\":{},\"gguf_path\":\"{}\",\"version\":{},\"tensor_count\":{},\"metadata_kv_count\":{},\"estimated_index_bytes\":{},\"prefetch_bytes\":{},\"io_chunk_bytes\":{},\"use_mmap\":{},\"use_pinned_staging\":{}}}",
            load_only,
            gguf_path.display(),
            header.version,
            header.tensor_count,
            header.metadata_kv_count,
            header.estimated_index_bytes(),
            plan.prefetch_bytes,
            plan.io_chunk_bytes,
            plan.use_mmap,
            plan.use_pinned_staging
        );
        return Ok(());
    }

    println!("gguf_path={}", gguf_path.display());
    println!("load_only={}", load_only);
    println!("version={}", header.version);
    println!("tensor_count={}", header.tensor_count);
    println!("metadata_kv_count={}", header.metadata_kv_count);
    println!("estimated_index_bytes={}", header.estimated_index_bytes());
    println!("prefetch_bytes={}", plan.prefetch_bytes);
    println!("io_chunk_bytes={}", plan.io_chunk_bytes);
    println!("use_mmap={}", plan.use_mmap);
    println!("use_pinned_staging={}", plan.use_pinned_staging);

    Ok(())
}
