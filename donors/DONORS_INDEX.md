# Donor Index (gfxATOM-Rust)

Scope: external donor/reference repos cloned under `gfxATOM-Rust/donors/`.

## Execution status (2026-05-16)

- Approved extraction+archive wave executed for one-concern/high-ROI donors.
- Extraction manifests written under:
  - `/home/local/ai/build/kernels/canonical/manifests/donor_extract_manifest_<repo>.json`
- Consolidated execution log:
  - `/home/local/ai/build/kernels/canonical/manifests/donor_extraction_execution_log.json`
- Harvest destination:
  - `/home/local/ai/build/kernels/canonical/harvested/donor_unique_features/`
- Archived repos moved to:
  - `/home/local/ai/projects/.archived/repos/`

Archived after successful extraction:
- `KVCache-Factory`
- `NautilusQuant`
- `QAQ-KVCacheQuantization`
- `RateQuant`
- `delta-k-quantization`
- `kv-cache-quantization`
- `norm-separated-quantization`
- `rust-llama.cpp`
- `turbo-quant`
- `vLLMTurboQuantKVCacheCPU`
- `vllm_backend`
- `wobble-quant-cache`
- `R-KV`
- `ik_llama.cpp`
- `llama-swap`
- `llm-compressor`
- `quant.cpp`
- `rvllm`
- `sardeenz`
- `smg`
- `tair-kvcache`

## Completed assimilations since the last index refresh

| Donor | Assimilation result | Integration surface | Notes |
|---|---|---|---|
| `llama-swap` | Hot-swap routing/lifecycle capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Extracted model aliasing, groups, TTL unload, request filters, config reload, and direct upstream passthrough as explicit runtime capability flags. |
| `sardeenz` | Multi-model packing / GPU residency capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Extracted dynamic load/unload, multi-model packing, multi-GPU distribution, kvcached sharing, sleep mode, move operations, and GPU memory telemetry as explicit runtime capability flags. |
| `tair-kvcache` | Storage orchestration / capacity-management capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured distributed pooling, dynamic multilevel caching, metadata management, capacity management, prefix/sliding-window/KV matching, two-phase writes, async eviction, and trace-replay optimization as read-only runtime capability flags. |
| `llm-compressor` | Quantization pipeline capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured model-free PTQ, compressed-tensors format, weight/activation/KV/attention pipelines, disk offloading, and distributed calibration as read-only runtime capability flags. |
| `rvllm` | Serving benchmark capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured Rust-native GPU serving, pure JAX TPU serving, cross-device benchmarking, single graph capture, FP8 channelscale epilogues, and benchmark telemetry as read-only runtime capability flags. |
| `vllm` / `vllm-omni` | Runtime family capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured multimodal serving, omni modality, and hardware plugin interface metadata as read-only runtime capability flags without changing backend execution behavior. `vllm-ascend` deleted prior to archive (no unique assimilation value). |
| `rvllm` | KV control-plane contract imports | `gfxATOM-Rust/crates/rs_kv_quant_contracts/src/lib.rs`, `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs` | Captured transfer, prefetch, eviction, step-graph, precomputed-asset, and decode-telemetry contract shapes as reusable shared KV types. |
| `rvllm` | Graph pool contract imports | `gfxATOM-Rust/crates/rs_kv_quant_contracts/src/lib.rs`, `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs` | Captured metadata-layout hashing, graph fingerprints, captured-graph lookup, and drift rejection as reusable replay contracts. |
| `rvllm` | Radix snapshot contract imports | `gfxATOM-Rust/crates/rs_kv_quant_contracts/src/lib.rs`, `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs` | Captured compact pre-order radix snapshots, node metadata, and bincode round-trips as reusable prefix-cache export contracts. |
| `smg` | Dual-hash routing helpers | `gfxATOM-Rust/crates/rs_kv_quant_contracts/src/lib.rs`, `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs` | Captured content-hash and request-hash helpers for cache-aware prefix routing. |
| `smg` | Prefix-match contract | `gfxATOM-Rust/crates/rs_kv_quant_contracts/src/lib.rs`, `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs` | Captured tenant IDs, prefix-match results, and hit-ratio helpers for routing bookkeeping. |
| `quant.cpp` | Delta-k KV delta compression profile | `gfxATOM-Rust/python/wave1_donor_adapters.py`, `gfxATOM-Rust/python/kv_policy_arbiter.py`, `gfxATOM-Rust/tests/test_kv_policy_arbiter.py`, `gfxATOM-Rust/tests/test_wave1_donor_adapters.py` | Captured key-delta compression as an explicit wave-1 policy lane with a dedicated profile helper and tests. |
| `wobble-quant-cache` | Variance-weighted KV allocation profile | `gfxATOM-Rust/python/wave1_donor_adapters.py`, `gfxATOM-Rust/python/kv_policy_arbiter.py`, `gfxATOM-Rust/tests/test_kv_policy_arbiter.py`, `gfxATOM-Rust/tests/test_wave1_donor_adapters.py` | Captured variance-aware bit allocation as a compact policy lane with deterministic profiling and tests. |
| `QAQ-KVCacheQuantization` | Quality-adaptive KV quantization profile | `gfxATOM-Rust/python/wave1_donor_adapters.py`, `gfxATOM-Rust/python/kv_policy_arbiter.py`, `gfxATOM-Rust/tests/test_kv_policy_arbiter.py`, `gfxATOM-Rust/tests/test_wave1_donor_adapters.py` | Captured quality-targeted KV quantization as a compact policy lane with a target-quality profile and tests. |
| `NautilusQuant` | Golden-ratio KV profile | `gfxATOM-Rust/python/wave1_donor_adapters.py`, `gfxATOM-Rust/python/kv_policy_arbiter.py`, `gfxATOM-Rust/tests/test_kv_policy_arbiter.py`, `gfxATOM-Rust/tests/test_wave1_donor_adapters.py` | Captured deterministic golden-ratio geometry as a reproducible policy lane with tests. |
| `ik_llama.cpp` | Layer offloading runtime profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured CPU/hybrid layer-offloading support as a minimal runtime capability flag without changing execution behavior. |
| `llama.cpp` | Stock upstream llama.cpp baseline | `/home/local/ai/projects/donors/llama.cpp` | Kept separate from `llama.cpp-1-bit-turbo` so upstream GGUF/profile harvesting stays clean and replayable. |
| `sglang-1-bit-turbo autoquant` | Observer snapshot parity | `gfxATOM-Rust/crates/rs_autoquant_policy/src/lib.rs`, `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs` | Captured observer-snapshot round-trip coverage for the autoquant policy crate and validation harness. |
| `gfxGRAPH` | Graph bridge capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured shape bucketing, validation, conditional nodes, and nested capture as explicit runtime capabilities. |

## Previously completed assimilation waves

| Wave | Result | Integration surface | Notes |
|---|---|---|---|
| `wave-1` | RateQuant / TurboQuant / norm-separated guardrail / CPU bench lane | `gfxATOM-Rust/python/kv_policy_arbiter.py`, `gfxATOM-Rust/python/wave1_donor_adapters.py`, `gfxATOM-Rust/scripts/wave1_kv_policy_canary.py`, `gfxATOM-Rust/tests/test_kv_policy_arbiter.py`, `gfxATOM-Rust/tests/test_wave1_donor_adapters.py`, `gfxATOM-Rust/tests/test_wave1_canary_cli.py` | Added gated policy selection, reverse-waterfilling bit allocation, TurboQuant split profiles, low-bit safety guardrails, and a CPU benchmark lane for parity checks. |
| `wave-2` | Adaptive policy recommendation lane | `gfxATOM-Rust/python/wave2_adaptive_policy.py`, `gfxATOM-Rust/scripts/wave1_kv_policy_canary.py`, `gfxATOM-Rust/tests/test_wave2_adaptive_policy.py` | Added advisory runtime-signal recommendations for baseline/ratequant/deltak/wobble/nautilus. |
| `wave-3` | Runtime adaptive recommendation profile field | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Exposed adaptive recommendation metadata in Python/Rust runtime contracts. |
| `wave-4` | Consul/Kong registration lane | `/home/local/ai/consul/services/inference.json`, `/home/local/ai/consul/services/api.json`, `ENCOM/servers/gateway/kong.yaml`, `console/port-registry.yml`, `tests/python/test_generate_consul_env.py` | Registered canonical `gfxatom-rust` and `api-gfxatom` service surfaces and locked the env/port mapping. |
| `wave-9` | `llama-swap` routing/lifecycle capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured hot swap, aliases, groups, TTL unload, request filters, config reload, and direct passthrough as runtime capabilities. |
| `wave-10` | `sardeenz` packing/residency capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured dynamic packing, multi-GPU distribution, kvcached sharing, sleep mode, move operations, and GPU memory telemetry as runtime capabilities. |
| `wave-11` | `quant.cpp` compact runtime capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured CPU-only runtime, on-demand download, Ollama-style CLI, OpenAI-compatible server, progressive KV compression, and full-document mode as compact-runtime capabilities. |
| `wave-12` | `vllm` FP8 KV-cache capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured FP8 KV-cache detail flags for per-tensor/per-head scales, calibration, and non-fused quantized attention. |
| `wave-13` | `R-KV` decode-time KV compression policy | `gfxATOM-Rust/python/kv_policy_arbiter.py`, `gfxATOM-Rust/python/wave1_donor_adapters.py`, `gfxATOM-Rust/tests/test_kv_policy_arbiter.py`, `gfxATOM-Rust/tests/test_wave1_donor_adapters.py`, `gfxATOM-Rust/docs/cli-args-and-feature-flags.md` | Captured redundancy-aware decode-time compression as a guarded policy family with observation-buffer, lambda, redundancy-window, and top-k parameters. |
| `wave-14` | `tair-kvcache` storage orchestration capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured distributed memory pooling, dynamic multilevel caching, metadata management, capacity management, prefix/sliding-window/KV matching, two-phase writes, async eviction, and trace replay optimization as runtime capability flags. |
| `wave-15` | `llm-compressor` quantization pipeline capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured model-free PTQ, compressed-tensors format, weight/activation/KV/attention pipelines, disk offloading, and distributed calibration as runtime capability flags. |
| `wave-16` | `rvllm` serving benchmark capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured Rust-native GPU serving, pure JAX TPU serving, cross-device benchmarking, single graph capture, FP8 channelscale epilogues, and benchmark telemetry as runtime capability flags. |
| `wave-30` | `vllm` / `vllm-omni` runtime family capability profile | `gfxATOM-Rust/python/engine_runtime_profile.py`, `gfxATOM-Rust/crates/rs_atom_engine_profile/src/lib.rs`, `gfxATOM-Rust/tests/test_engine_runtime_profile_schema.py` | Captured multimodal serving, omni modality, and hardware plugin interface metadata as runtime capability flags. `vllm-ascend` deleted (no unique assimilation value). |
| `wave-31` | `smg` positional-index contract (archived) | `gfxATOM-Rust/crates/rs_kv_quant_contracts/src/lib.rs`, `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs` | Added `PositionalIndexKey`, `PositionalIndexEntry`, `PositionalIndexError`, and `PositionalIndexResult<T>`. `smg` donor archived post-assimilation. |
| `wave-17` | `rvllm` KV control-plane contract imports | `gfxATOM-Rust/crates/rs_kv_quant_contracts/src/lib.rs`, `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs` | Captured transfer, prefetch, eviction, step-graph, precomputed-asset, and decode-telemetry contract shapes as reusable KV types. |
| `wave-18` | `rvllm` graph capture pool imports | `gfxATOM-Rust/crates/rs_kv_quant_contracts/src/lib.rs`, `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs` | Captured metadata-layout hashing, graph fingerprints, captured-graph lookup, and drift rejection as reusable replay contracts. |
| `wave-19` | `rvllm` radix snapshot contract imports | `gfxATOM-Rust/crates/rs_kv_quant_contracts/src/lib.rs`, `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs` | Captured compact pre-order radix snapshots, node metadata, and bincode round-trips as reusable prefix-cache export contracts. |
| `wave-20` | `smg` dual-hash routing helpers | `gfxATOM-Rust/crates/rs_kv_quant_contracts/src/lib.rs`, `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs` | Captured deterministic content-hash and chunked request-hash helpers for prefix routing. |
| `wave-21` | `smg` prefix-match contract | `gfxATOM-Rust/crates/rs_kv_quant_contracts/src/lib.rs`, `gfxATOM-Rust/crates/rs_kv_validation_harness/src/lib.rs` | Captured tenant IDs, prefix-match results, and hit-ratio helpers for routing bookkeeping. |

## Other completed wave results

| Wave | Result | Integration surface | Notes |
|---|---|---|---|
| `wave-2-preservation` | Donor-wave executor targeting plus Chatterbox/LFM preservation dry-runs | `scripts/apply_wip_kernel_queue.py`, canonical harvested donor lanes under `/home/local/ai/build/kernels/canonical/harvested/` | Added focused repo-pattern targeting, preservation lanes, and canonical harvest summaries for audio-family retention. |
| `tranche-17` | Donor deep-harvest wave and ranked extraction manifests | `/home/local/ai/build/kernels/canonical/manifests/`, `DONOR_INTELLIGENCE_NOTES.md` | Locked wave focus to `kv_attention_quant`, generated ranked extraction entries, and refreshed donor intelligence notes. |

Legend:
- **Claimed focus** = what repo README/docs explicitly claims.
- **Best-at (for us)** = where this donor appears most valuable for fusion work.

| Donor | Claimed focus | Best-at (for gfxATOM-Rust) | Notes |
|---|---|---|---|
| `vllm` | “Easy, fast, and cheap LLM serving” | Baseline serving scheduler + API/runtime parity target | Primary upstream reference for serving behavior and benchmarks. |
| `vllm-omni` | Omni-modality serving across CUDA/ROCm/NPU/XPU | Multimodal runtime patterns, diffusion/audio serving surfaces | Useful for future omni extensions beyond text-only engine path. |
| `vllm-ascend` | vLLM plugin for Ascend hardware | Plugin split architecture, hardware-specific backend separation | Good reference for clean hardware-plugin boundaries. |
| `vllm_backend` | Triton backend for vLLM AsyncEngine | Triton integration model and deployment packaging | Useful if we expose vLLM-compatible inference via Triton. |
| `llm-compressor` | Quantization/transforms (weights/acts/KV/attention) for vLLM deployment | Quant pipeline design and compressed-tensors compatibility ideas | Strong donor for quant tooling flow and artifact formats. |
| `KVCache-Factory` | Unified framework for KV cache compression (PyramidKV/SnapKV/H2O/StreamingLLM) | KV compression method matrix and cross-model experiments | High-value donor for pluggable KV policies and benchmarking. |
| `R-KV` | Decode-time KV compression cluster, “shrink cache keep accuracy” | Compression-aware scheduling/memory pooling integration patterns | Directly relevant to SGLang-side KV compression integration. |
| `vLLMTurboQuantKVCacheCPU` | CPU-only TurboQuant KV cache evaluation framework for vLLM | Benchmarking harness and metrics framing for simulated KV compression | Good reference for comparing baseline vs TurboQuant-style strategies. |
| `delta-k-quantization` | Group-size-robust KV quantization via closed-loop differential encoding | 2-bit K/V compression policy ideas and group-size robustness analysis | Strong donor for low-bit differential KV codecs. |
| `RateQuant` | Optimal mixed-precision KV cache quantization via rate-distortion theory | Bit allocation / mixed-precision policy selection | Excellent fit for policy-layer bit budgeting. |
| `NautilusQuant` | Deterministic orthogonal KV-cache quantization via golden-ratio geometry | Deterministic transform ideas and quantization geometry | Good candidate for reproducible codec experimentation. |
| `norm-separated-quantization` | Training-free fix for KV cache INT4 failures via norm separation | Failure-mode mitigation for INT4-style codecs | Strong donor for “don’t catastrophically break” guardrails. |
| `Higman-sims-quant` | Lattice-RSN / E8-space KV-cache compression framework | Ultra-lossless / lattice-codec experimentation | Useful as a high-end theoretical and codec-space reference. |
| `wobble-quant-cache` | Adaptive KV cache quantization that allocates bits by variance | Per-dimension adaptive bit allocation heuristics | Good fit for dimension-aware mixed precision policies. |
| `QAQ-KVCacheQuantization` | Quality-adaptive KV quantization; claims ~10x compression with minimal loss | Adaptive precision policy ideas for long-context KV | Good research donor for quality-aware bit allocation logic. |
| `turbo-quant` | Rust TurboQuant/PolarQuant/QJL for embeddings + KV compression | Rust-native low-bit vector/KV quant algorithms | Very strong donor for Rust-first quantization core. |
| `kv-cache-quantization` | Adaptive precision KV with freshness window + INT8 aging | Simple recency-based precision scheduling heuristic | Useful as lightweight baseline policy and ablation reference. |
| `tair-kvcache` | Alibaba distributed KVCache system with multi-level caching/memory pooling | Distributed KV metadata, control-plane design, capacity management | Strong systems donor for multi-tier KV orchestration. |
| `quant.cpp` | “SQLite of LLM inference”, GGUF runtime, claims 6.4x KV compression | Minimal C/C++ embedding + compact runtime integration approach | Useful for lean runtime API design and embeddability. |
| `ik_llama.cpp` | llama.cpp fork claiming stronger CPU/hybrid perf + quant variants | CPU/hybrid execution strategy, quant formats, MoE hybrid offload tactics | Useful when optimizing CPU fallback and hybrid placement. |
| `rust-llama.cpp` | Rust bindings over llama.cpp | Rust FFI/bindings patterns for llama.cpp ecosystem | Useful for adapter layer patterns, less for core kernels. |
| `rvllm` | Rust+CUDA GPU and JAX+XLA TPU inference engine | Rust serving architecture and GPU/TPU split design | Good architecture donor for Python-minimized serving path. |
| `llama-swap` | Hot-swap model gateway for OpenAI-compatible backends | Multi-engine routing and model swap control-plane UX | Useful for front-door gateway/routing semantics. |
| `smg` | High-performance model-routing gateway | Worker lifecycle + traffic routing + enterprise gateway controls | Useful for orchestrator features around engine fleet management. |
| `sardeenz` | PoC multi-model packing on GPU(s), built on vLLM + kvcached | Multi-model residency and GPU utilization strategies | Useful for model packing and placement policies. |

## Priority donor set (first-pass assimilation)

1. `RateQuant`, `delta-k-quantization`, `turbo-quant`, `R-KV`
2. `KVCache-Factory`, `QAQ-KVCacheQuantization`, `wobble-quant-cache`, `norm-separated-quantization`
3. `tair-kvcache`, `vllm`, `llm-compressor`, `rvllm`
4. `vLLMTurboQuantKVCacheCPU`, `quant.cpp`, `ik_llama.cpp`, `NautilusQuant`, `Higman-sims-quant`
5. `llama-swap`, `smg`, `sardeenz`, `vllm_backend`, `vllm-omni`, `vllm-ascend`, `rust-llama.cpp`, `kv-cache-quantization`

## Assimilation notes

- Treat README claims as **unverified** until measured in our RDNA2/ROCm test matrix.
- For quant donors, preserve side-by-side parity tests: perplexity, long-context retention, prefix reuse stability, and tokens/sec.
- For systems donors (gateway/control-plane), mine architecture and contract patterns before code transplantation.
