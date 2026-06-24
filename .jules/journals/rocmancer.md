# Rocmancer Journal

## Run 1: Enable Triton MoE for gfx1030

- **What was changed**: Added `gfx.startswith("gfx103")` to the `self.use_triton` assignment condition in `atom/model_ops/moe.py`.
- **Why it matters on gfx1030**: RDNA2 consumer cards previously fell back to slower non-Triton paths. Enabling Triton kernels opens up performance opportunities for MoE models on this architecture.
- **What was learned**: The codebase relies heavily on the `get_gfx()` from `aiter.jit.utils.chip_info` to route hardware-specific kernel logic. Testing logic required mocking specific modules like `aiter` as the local test environment didn't have access to the AMD-specific binaries.
- **What remains risky**: Not explicitly tested on hardware, so Triton kernel availability/compatibility needs field verification. The underlying `fused_moe_triton.py` might still have implicit warp/wavefront size assumptions that differ from `gfx94`.
- **Next steps**: Explore `fused_moe_triton.py` and `attention_mla.py` for wavefront (32 vs 64) or LDS tuning specifically for `gfx1030`, or refine KV cache allocations.
# rocmancer journal

## 2026-05-17 — Wave 34G llama.cpp family catalog

- Added `crates/rs_gguf_loader_core/src/llama_cpp_family.rs` to derive family groupings, attention traits, graph traits, and model-role hints from the llama.cpp donor surface.
- Added `gguf-family-import` to write `inventory/llama_cpp_family_catalog.json`.
- Wired `gguf-benchmark` to embed the family catalog and emit a selected family id for each benchmarked model.

### What I learned

- The maintained llama.cpp fork already includes Jina-family additions, and the additive family catalog keeps them visible instead of collapsing them into upstream buckets.
- A small amount of source-marker scanning from `llama-model.cpp` and `llama-graph.cpp` is enough to capture the attention features we care about right now (`moe`, `pooling_type`, `softcapping`, `vision`).

### Remaining risk

- The family map is still heuristic; if upstream llama.cpp adds a new attention family with a unique routing policy, we will need to add a new normalized family bucket instead of relying on the fallback name.
- The catalog currently tracks source markers rather than a full semantic attention policy graph; that’s enough for selection and import, but not yet a complete runtime router.

## 2026-05-17 — Wave 34F stock llama.cpp donor baseline

- Cloned a separate upstream llama.cpp donor into `/home/local/ai/projects/donors/llama.cpp`.
- Kept the special `llama.cpp-1-bit-turbo` fork isolated so upstream capability harvesting can be refreshed without mixing in fork-specific changes.

### What I learned

- A clean donor split matters: the upstream baseline is now the right source for model/profile harvesting, while the fork remains the AMD-specific execution target.

### Remaining risk

- The donor tree still needs a small refresh workflow if upstream llama.cpp changes faster than we update the snapshot/import tool.

## 2026-05-17 — Wave 34E llama.cpp capability surface importer

- Added `crates/rs_gguf_loader_core/src/llama_cpp_surface.rs` to snapshot llama.cpp GGUF capabilities from donor source.
- Added `gguf-profile-import` to write the normalized surface into `inventory/llama_cpp_gguf_surface.json`.
- Wired `gguf-benchmark` to carry the imported surface and gate llama.cpp profile availability by architecture + quantization support.

### What I learned

- llama.cpp does not hand us a ready-made profile catalog, but its `llama-arch.cpp` and `llama-model-loader.cpp` source maps are rich enough to reconstruct a stable capability surface.
- Normalizing HF-style architecture names is necessary if we want the imported llama.cpp surface to line up with actual model configs instead of just source identifiers.

### Remaining risk

- The current importer is source-surface based rather than a full runtime introspector; if llama.cpp moves capability metadata into another file, the source paths will need refresh logic.
- Next step should expand from surface import to a broader model/profile registry for more than the qwen benchmark pair.

## 2026-05-17 — Wave 34D Rust GGUF comparison runner

- Moved the GGUF comparison workflow into Rust with `crates/rs_gguf_loader_core/src/bin/gguf-benchmark.rs`.
- Kept the Python benchmark file as a thin launcher only; the actual comparison logic now lives in Rust.
- Captured real qwen GGUF results in `benchmarks/gguf_load_comparison.json`.

### What I learned

- The Rust load-only path is effectively instantaneous compared with llama.cpp single-turn startup on these tiny qwen GGUFs.
- For TurboRotor footprint reporting, the config directory must be resolved from the sibling base model directory, not from the GGUF folder itself.

### Remaining risk

- The benchmark currently only covers the qwen GGUF pair; a broader quant sweep can be added on top of the Rust runner next.
- If we want model-family discovery across more registries, the Rust runner should get a tiny path registry rather than hard-coded qwen discovery.

## 2026-05-17 — Wave 34A GGUF comparative assimilation tranche

- Implemented `python/gguf_pipeline_comparator.py` to convert GGUF integration strategy from ad-hoc notes into executable comparative evidence across SGLang, ATOM, and llama.cpp.
- Added a score-based ranking and deterministic assimilation-step synthesis for ATOM backend cutover planning.
- Added `benchmarks/compare_gguf_pipelines.py` for repeatable JSON report generation and quick terminal ranking output.
- Added `tests/test_gguf_pipeline_comparator.py` to lock signal detection + synthesis behavior.

### What I learned

- We already had parity notes, but they were mostly static and partially TODO-shaped; converting this into code gives a reusable gate before every assimilation tranche.
- Backend replacement planning is safer when GGUF loader/path differences are measured and then translated to explicit Rust modules.

### Remaining risk

- Current comparator is static source-signal based; it does not yet execute live model-load timings.
- Next tranche should add timing probes around parser/load/init for all three engines, then fold those metrics into assimilation prioritization.

## 2026-05-17 — Wave 34B GGUF loader core assimilation

- Implemented `crates/rs_gguf_loader_core` as ATOM-side Rust GGUF loader foundation:
  - strict v3 header parser from bytes/path
  - explicit typed failure modes (magic/version/truncation/io)
  - deterministic prefetch/chunk/mmap/pinned-staging load-plan synthesis
- Added `gguf-plan` CLI binary for fast operational inspection of GGUF files and generated load plans.
- Registered crate in workspace and added unit tests for parser correctness and planner behavior.

### What I learned

- The biggest immediate gap to close against llama.cpp is not only quant kernels but front-end GGUF load-path determinism and staging policy.
- Encoding load-planning policy in Rust now gives us a stable seam to plug into ATOM backend cutover gates without waiting on full Python orchestration rewrite.

### Remaining risk

- This tranche parses only the fixed header; tensor/metadata-table walking is next.
- Need direct Python/Rust bridge for this crate so runtime can use it without subprocess boundaries.

## 2026-05-17 — Wave 34C E2E + benchmark tranche

- Added `tests/test_universal_kv_broker_e2e.py` to lock model-agnostic broker contracts:
  - cross-model materialization shape contract
  - forced spill/restore behavior under pooled multi-model pressure
- Added `benchmarks/benchmark_universal_kv_broker.py` with CLI JSON output for hot-tier hit rate, spill latency percentile metrics, and effective context expansion estimate.
- Promoted storage-orchestration defaults in runtime profiles (Python + Rust) to reflect current broker architecture baseline:
  - distributed memory pooling
  - dynamic multilevel caching
  - kv matching
  - async eviction

### What I learned

- The most actionable cross-model contract assertion is shape-valid/finite materialization from a model-agnostic block, not semantic equivalence.
- The benchmark harness benefits from small deterministic synthetic traces that force tier transitions quickly; this keeps integration feedback loops short.

### Remaining risk

- E2E still uses synthetic tensors; next tranche should add real GGUF-backed request traces and decode-phase timing under sustained load.
