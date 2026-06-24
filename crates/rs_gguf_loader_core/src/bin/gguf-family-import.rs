use std::fs;
use std::path::{Path, PathBuf};

use rs_gguf_loader_core::llama_cpp_family::load_llama_cpp_family_catalog;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut source_root = PathBuf::from("/home/local/ai/projects/donors/llama.cpp-1-bit-turbo");
    let mut output_path = PathBuf::from("/home/local/ai/build/wip/gfxATOM-Rust/inventory/llama_cpp_family_catalog.json");

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--source-root" => {
                source_root = PathBuf::from(args.next().ok_or("--source-root requires a path")?);
            }
            "--output" => {
                output_path = PathBuf::from(args.next().ok_or("--output requires a path")?);
            }
            value if value.starts_with("--") => {
                return Err(format!("unknown flag: {}", value).into());
            }
            _ => {}
        }
    }

    let catalog = load_llama_cpp_family_catalog(Path::new(&source_root))?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, serde_json::to_string_pretty(&catalog)? + "\n")?;
    println!("{}", serde_json::to_string_pretty(&catalog)?);
    println!("saved: {}", output_path.display());
    Ok(())
}
