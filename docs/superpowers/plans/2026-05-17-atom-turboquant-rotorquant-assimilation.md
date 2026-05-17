# ATOM TurboQuant & RotorQuant Assimilation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ATOM ingest the useful TurboQuant and RotorQuant behavior from SGLang, vLLM, and llama.cpp through a Rust-first capability and routing layer.

**Architecture:** ATOM keeps a canonical Rust quant surface registry and runtime profile. Python adapters stay thin: they parse engine-specific config, call into Rust for codec/fallback decisions, and only retain compatibility shims where donor code still owns a surface. SGLang remains the source of truth for KV semantics, vLLM contributes the best TurboQuant runtime backend and preset naming, and llama.cpp contributes ggml low-bit tensor/block layout details.

**Tech Stack:** Rust (`rs_atom_engine_profile`, `rs_kv_quant_contracts`, `rs_kv_codec_adapters`, `rs_kv_validation_harness`, `rs_turboquant_codec`, new RotorQuant crate if needed), Python shims under `gfxATOM-Rust/python`, `pytest`, `cargo test`.

---

### Task 1: Canonicalize the cross-engine quant surface in Rust

**Files:**
- Modify: `crates/rs_atom_engine_profile/src/lib.rs`
- Modify: `crates/rs_kv_quant_contracts/src/lib.rs`
- Modify: `crates/rs_kv_codec_adapters/src/lib.rs`
- Modify: `crates/rs_kv_validation_harness/src/lib.rs`
- Test: `crates/rs_kv_validation_harness/src/turboquant_tests.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn runtime_profile_reports_tq_and_rq_support() {
    let profile = EngineRuntimeProfile {
        supports_atom_backend: true,
        supports_atom_attention: true,
        supports_atom_kv_quant: true,
        supports_atom_rocm_telemetry: true,
        supports_atom_fallback: true,
        supports_automatic_prefix_caching: true,
        supports_radix_cache: true,
        supports_lmcache_connector: true,
        supports_kv_events: true,
        supports_fp8_kv_cache: true,
        supports_fp8_kv_per_tensor_scales: true,
        supports_fp8_kv_per_head_scales: false,
        supports_kv_scale_calibration: true,
        supports_quantized_attention_fusion: true,
        supports_turboquant_kv: true,
        supports_rotorquant_kv: true,
        supports_eagle3: true,
        supports_medusa: false,
        supports_ngram_speculation: false,
        supports_phantom: false,
        supports_phantom_x: false,
        supports_disaggregated_prefill: false,
        supports_disaggregated_decode: false,
        supports_prefix_aware_attention: true,
        supports_content_addressed_cache: true,
        supports_position_independent_cache: true,
        supports_model_hot_swap: false,
        supports_model_aliases: false,
        supports_model_groups: false,
        supports_ttl_unload: false,
        supports_request_filters: false,
        supports_config_reload: false,
        supports_direct_model_passthrough: false,
        supports_dynamic_model_loading: false,
        supports_dynamic_model_unloading: false,
        supports_multi_model_packing: false,
        supports_multi_gpu_distribution: false,
        supports_kvcached_memory_sharing: false,
        supports_model_sleep_mode: false,
        supports_model_move_operations: false,
        supports_layer_offloading: false,
        supports_gpu_memory_telemetry: true,
        supports_cpu_only_runtime: false,
        supports_download_on_first_use: false,
        supports_ollama_style_cli: false,
        supports_openai_compatible_server: true,
        supports_progressive_kv_compression: true,
        supports_full_document_mode: false,
        supports_distributed_memory_pooling: false,
        supports_dynamic_multilevel_caching: true,
        supports_global_metadata_management: true,
        supports_capacity_management: true,
        supports_prefix_matching: true,
        supports_sliding_window_matching: true,
        supports_kv_matching: true,
        supports_two_phase_write: true,
        supports_async_eviction: true,
        supports_trace_replay_optimization: false,
        supports_model_free_ptq: false,
        supports_compressed_tensors_format: false,
        supports_weight_quantization_pipeline: true,
        supports_activation_quantization_pipeline: false,
        supports_kv_cache_quantization_pipeline: true,
    };

    assert!(profile.supports_turboquant_kv);
    assert!(profile.supports_rotorquant_kv);
}
```

- [ ] **Step 2: Run the failing test**

Run: `cargo test -p rs_atom_engine_profile runtime_profile_reports_tq_and_rq_support -v`

Expected: fail because the profile builder and/or validation helpers do not yet advertise the unified TQ/RQ surface.

- [ ] **Step 3: Implement the canonical mapping**

Add the minimal Rust mapping needed so ATOM can represent:
- SGLang TurboQuant KV: `tq2`, `tq3`, `tq4`
- SGLang RotorQuant KV: `rq3_planar`, `rq4_planar`
- vLLM TurboQuant presets: `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`
- llama.cpp low-bit donor surfaces: `tq1_0`, `tq2_0`, `block_planar3_0`, `block_planar4_0`, `block_iso3_0`, `block_iso4_0`

Keep the profile booleans fail-closed by default and only flip them on for a codec family that is explicitly supported.

- [ ] **Step 4: Run the Rust validation tests**

Run:
`cargo test -p rs_kv_codec_adapters -p rs_kv_validation_harness -p rs_atom_engine_profile`

Expected: pass with the new capability matrix and no regressions in the existing codec contract tests.

- [ ] **Step 5: Commit**

```bash
git add crates/rs_atom_engine_profile/src/lib.rs crates/rs_kv_quant_contracts/src/lib.rs crates/rs_kv_codec_adapters/src/lib.rs crates/rs_kv_validation_harness/src/lib.rs crates/rs_kv_validation_harness/src/turboquant_tests.rs
git commit -m "feat: canonicalize turboquant and rotorquant surfaces"
```

### Task 2: Move TurboQuant preset semantics into the Rust hot path

**Files:**
- Modify: `crates/rs_turboquant_codec/src/lib.rs`
- Modify: `crates/rs_turboquant_codec/src/polar.rs`
- Modify: `crates/rs_turboquant_codec/src/qjl.rs`
- Modify: `crates/rs_turboquant_codec/src/turbo.rs`
- Modify: `python/kv_codec_adapters.py`
- Modify: `python/sglang_backend_adapter.py`
- Test: `tests/test_sglang_autoquant_bridge.py`
- Test: `crates/rs_kv_validation_harness/src/turboquant_tests.rs`

- [ ] **Step 1: Write the failing test**

Add a Rust test that asserts the named vLLM-style presets round-trip to exact metadata:
```rust
#[test]
fn turboquant_presets_match_named_modes() {
    let qz = TurboQuantizer::new(128, 4, 32, 42).unwrap();
    assert_eq!(qz.bits(), 4);
    assert_eq!(qz.projections(), 32);
}
```

Add a Python test that asserts the backend plan for `tq2` and `tq4` always returns the same ordered chain:
`["turboquant", "triton", "fp16"]`.

- [ ] **Step 2: Run the failing test**

Run:
`cargo test -p rs_turboquant_codec turboquant_presets_match_named_modes -v`
`pytest -q tests/test_sglang_autoquant_bridge.py::test_registry_turboquant_factory_exposes_fallback_chain`

Expected: the named preset / chain checks fail before the metadata and adapter bridge are fully wired.

- [ ] **Step 3: Implement the preset bridge**

Teach the Rust codec and Python adapter to understand the concrete TurboQuant KV presets from vLLM and SGLang:
- `turboquant_k8v4`
- `turboquant_4bit_nc`
- `turboquant_k3v4_nc`
- `turboquant_3bit_nc`
- `tq1` / `tq2` / `tq3` / `tq4` / `tq8`

Keep the runtime chain Rust-owned and make Python a thin config parser and logger.

- [ ] **Step 4: Run the Rust and Python tests**

Run:
`cargo test -p rs_turboquant_codec -p rs_kv_validation_harness`
`pytest -q tests/test_sglang_autoquant_bridge.py tests/test_sglang_config_wrapper.py`

Expected: both suites pass and the plan summary still reports `turboquant -> triton -> fp16` for TurboQuant modes.

- [ ] **Step 5: Commit**

```bash
git add crates/rs_turboquant_codec/src/lib.rs crates/rs_turboquant_codec/src/polar.rs crates/rs_turboquant_codec/src/qjl.rs crates/rs_turboquant_codec/src/turbo.rs python/kv_codec_adapters.py python/sglang_backend_adapter.py tests/test_sglang_autoquant_bridge.py crates/rs_kv_validation_harness/src/turboquant_tests.rs
git commit -m "feat: route turboquant presets through rust metadata"
```

### Task 3: Assimilate RotorQuant block/layout behavior as a Rust module

**Files:**
- Create: `crates/rs_rotorquant_codec/Cargo.toml`
- Create: `crates/rs_rotorquant_codec/src/lib.rs`
- Create: `crates/rs_rotorquant_codec/src/planar.rs`
- Create: `crates/rs_rotorquant_codec/src/iso.rs`
- Create: `crates/rs_rotorquant_codec/src/error.rs`
- Modify: root Rust workspace manifest
- Modify: `crates/rs_kv_codec_adapters/src/lib.rs`
- Modify: `crates/rs_kv_validation_harness/src/lib.rs`

- [ ] **Step 1: Write the failing test**

Add Rust tests that assert the RotorQuant block types match the llama.cpp donor layouts:
```rust
#[test]
fn rotorquant_block_sizes_match_donor_layouts() {
    assert_eq!(std::mem::size_of::<BlockPlanar3_0>(), 48);
    assert_eq!(std::mem::size_of::<BlockPlanar4_0>(), 66);
    assert_eq!(std::mem::size_of::<BlockIso3_0>(), 48);
    assert_eq!(std::mem::size_of::<BlockIso4_0>(), 66);
}
```

- [ ] **Step 2: Run the failing test**

Run: `cargo test -p rs_rotorquant_codec rotorquant_block_sizes_match_donor_layouts -v`

Expected: fail until the module exists and the exact block layout is encoded in Rust.

- [ ] **Step 3: Implement the block/layout module**

Port the planar and iso donor layouts into Rust with the same packed semantics as the llama.cpp ggml blocks. Keep the module limited to:
- block structs
- alias normalization
- pack/unpack helpers
- deterministic tests

Do not add a new public engine abstraction if the module can stay small and focused.

- [ ] **Step 4: Validate the RotorQuant module**

Run:
`cargo test -p rs_rotorquant_codec`
`cargo test -p rs_kv_validation_harness rotorquant`

Expected: block sizes, alias mapping, and pack/unpack round-trips pass.

- [ ] **Step 5: Commit**

```bash
git add crates/rs_rotorquant_codec Cargo.toml crates/rs_kv_codec_adapters/src/lib.rs crates/rs_kv_validation_harness/src/lib.rs
git commit -m "feat: add rotorquant rust layout module"
```

### Task 4: Wire ATOM runtime routing, telemetry, and fallback gating

**Files:**
- Modify: `python/engine_runtime_profile.py`
- Modify: `crates/rs_atom_engine_profile/src/lib.rs`
- Modify: `python/sglang_backend_adapter.py`
- Modify: `python/sglang_autoquant_bridge.py`
- Modify: `crates/rs_kv_validation_harness/src/lib.rs`
- Modify: `docs/AMD_ENGINE_SUPPORT_AUDIT.md`
- Modify: `CHANGELOG.md`
- Test: `tests/test_engine_runtime_profile_schema.py`

- [ ] **Step 1: Write the failing test**

Add a profile schema test that asserts the ATOM profile reports:
- TurboQuant enabled for `tq2/tq3/tq4`
- RotorQuant enabled for `rq3_planar/rq4_planar`
- fallback chain metadata present in the serialized config

- [ ] **Step 2: Run the failing test**

Run: `pytest -q tests/test_engine_runtime_profile_schema.py`

Expected: fail until the profile payload includes the unified routing and telemetry flags.

- [ ] **Step 3: Implement the routing and telemetry plumbing**

Make the runtime profile and Python adapters emit:
- supported quant family
- preferred backend
- fallback backend
- ultimate fallback
- ROCm/hardware gating status
- compression ratio / fallback event telemetry

Keep the logic fail-closed: unsupported surfaces must route to Triton or FP16 rather than silently claiming TurboQuant/RotorQuant support.

- [ ] **Step 4: Run the full validation slice**

Run:
`cargo test -p rs_atom_engine_profile -p rs_kv_validation_harness -p rs_kv_codec_adapters -p rs_turboquant_codec -p rs_rotorquant_codec`
`pytest -q tests/test_engine_runtime_profile_schema.py tests/test_sglang_autoquant_bridge.py tests/test_sglang_config_wrapper.py`

Expected: all pass with the new profile, telemetry, and fallback wiring.

- [ ] **Step 5: Commit**

```bash
git add python/engine_runtime_profile.py python/sglang_backend_adapter.py python/sglang_autoquant_bridge.py crates/rs_atom_engine_profile/src/lib.rs crates/rs_kv_validation_harness/src/lib.rs tests/test_engine_runtime_profile_schema.py docs/AMD_ENGINE_SUPPORT_AUDIT.md CHANGELOG.md
git commit -m "feat: wire atom turboquant rotorquant routing"
```

### Task 5: Final integration review

**Files:**
- Review: all changed files from Tasks 1-4
- Update: `docs/CROSS_ENGINE_PARITY_ANALYSIS.md` if the new runtime surface changes any matrix rows

- [ ] **Step 1: Run the targeted test suite**

Run:
`cargo test`
`pytest -q tests/test_engine_runtime_profile_schema.py tests/test_sglang_autoquant_bridge.py tests/test_sglang_config_wrapper.py`

- [ ] **Step 2: Check for plan gaps**

Confirm the final code paths cover:
- SGLang TurboQuant and RotorQuant donor semantics
- vLLM TurboQuant runtime backend semantics
- llama.cpp TQ1/TQ2 and RotorQuant block/layout semantics
- Rust-owned fallback and telemetry decisions

- [ ] **Step 3: Update cross-engine docs**

Only if the matrix changed: update `docs/CROSS_ENGINE_PARITY_ANALYSIS.md` so it matches the final ATOM surface.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete atom turboquant rotorquant assimilation"
```
