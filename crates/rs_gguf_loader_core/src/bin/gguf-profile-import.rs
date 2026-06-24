use std::path::PathBuf;

use rs_gguf_loader_core::llama_cpp_surface::{load_llama_cpp_surface, LlamaCppCapabilitySurface};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let source_root = PathBuf::from(
        args.next()
            .ok_or("usage: gguf-profile-import <llama.cpp-root> [output.json]")?,
    );
    let output_path = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/local/ai/build/wip/gfxATOM-Rust/inventory/llama_cpp_gguf_surface.json")
    });

    let surface: LlamaCppCapabilitySurface = load_llama_cpp_surface(&source_root)?;
    std::fs::create_dir_all(
        output_path
            .parent()
            .ok_or("output path missing parent directory")?,
    )?;
    std::fs::write(&output_path, serde_json::to_string_pretty(&surface)? + "\n")?;

    println!("source_root={}", surface.source_root);
    println!("architectures={}", surface.architectures.len());
    println!("quantization_labels={}", surface.quantization_labels.len());
    println!("loader_kv_keys={}", surface.loader_kv_keys.len());
    println!("saved={}", output_path.display());
    Ok(())
}
