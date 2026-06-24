# rs_gguf_loader_core

Rust crate in the gfxATOM submodule workspace. Rust-first GGUF header parser and load-plan core for ATOM backend assimilation — reads GGUF v3 headers, validates magic and version, and emits `GgufLoadPlan` instances for downstream prefetch and pinned-RAM staging.

## Navigation

- Package manifest: `Cargo.toml`
- Change history: `CHANGELOG.md`
- Canonical repository documentation: consult the nearest repository `docs/` directory and workspace-level architecture notes.

## Maintenance

Keep this README as a crate-local routing page. Put durable design details in canonical repository documentation and record crate-specific changes in `CHANGELOG.md`.
