# Donor Feature Integration & Optimization Sprint (ROCm/gfx1030)

## Scope

This sprint covers the extracted one-concern donors already harvested into:

- `/home/local/ai/build/kernels/canonical/harvested/donor_unique_features/`

Extracted donors in scope:

1. `KVCache-Factory`
2. `NautilusQuant`
3. `QAQ-KVCacheQuantization`
4. `RateQuant`
5. `delta-k-quantization`
6. `kv-cache-quantization`
7. `norm-separated-quantization`
8. `rust-llama.cpp`
9. `turbo-quant`
10. `vLLMTurboQuantKVCacheCPU`
11. `vllm_backend`
12. `wobble-quant-cache`

## Business Logic Collision Analysis (Before Integration)

### Collision domain A: KV bit-allocation policy overlap

Competing sources:

- `RateQuant` (rate-distortion)
- `wobble-quant-cache` (variance-aware per-dimension)
- `QAQ-KVCacheQuantization` (quality-adaptive search)
- `delta-k-quantization` (differential compression)

Potential collision:

- Multiple policy engines may independently choose incompatible bit layouts for the same KV block.

Resolution:

- Introduce one policy arbitration layer in `mb-kv-cache`:
  - choose exactly one `kv_policy_family` per request/session
  - enforce canonical policy outputs: `{k_bits, v_bits, group_size, calibration_profile}`

### Collision domain B: recency/freshness vs deterministic transforms

Competing sources:

- `kv-cache-quantization` (freshness-window aging)
- `NautilusQuant` (deterministic transform/LUT)
- `turbo-quant` (Rust low-bit transform primitives)

Potential collision:

- Aging policy can invalidate assumptions used by deterministic transform or decode-time calibration.

Resolution:

- Keep transforms deterministic and stable; apply aging only as an outer-tier eviction/precision downgrade policy.

### Collision domain C: runtime/deployment adapters

Competing sources:

- `vllm_backend` deployment patterns
- `rust-llama.cpp` adapter patterns

Potential collision:

- Runtime wrapper behavior can drift from engine-native scheduling/metrics contracts.

Resolution:

- Treat adapters as edge bindings only; do not let adapter code own quant/KV policy decisions.

## Clear Benefit Map by Extracted Donor

| Donor | Assimilated feature | Benefit | Collision risk | Feature flag |
|---|---|---|---|---|
| KVCache-Factory | Multi-policy KV benchmark harness | Rapid A/B policy comparison | Medium | `GFXATOM_KV_POLICY_FACTORY=1` |
| NautilusQuant | Deterministic transform/LUT quant ideas | Reproducible codec behavior | Medium | `GFXATOM_KV_NAUTILUS=1` |
| QAQ-KVCacheQuantization | Quality-adaptive quant scoring | Better quality under long context | Medium | `GFXATOM_KV_QAQ=1` |
| RateQuant | Rate-distortion bit budgeting | Principled mixed precision | Low | `GFXATOM_KV_RATEQUANT=1` |
| delta-k-quantization | Differential K/V compression | Strong low-bit compression lane | Medium | `GFXATOM_KV_DELTAK=1` |
| kv-cache-quantization | Freshness-window precision aging | Lightweight memory pressure control | Medium | `GFXATOM_KV_AGING=1` |
| norm-separated-quantization | INT4 failure guardrail | Safer low-bit rollout | Low | `GFXATOM_KV_NORM_GUARD=1` |
| rust-llama.cpp | Rust FFI adapter pattern | Cleaner Rust boundary for gguf adapters | Low | `GFXATOM_RUST_LLAMA_ADAPTER=1` |
| turbo-quant | Rust Turbo/QJL/Polar primitives | Fast Rust-native quant core | Low | `GFXATOM_KV_TURBOQUANT=1` |
| vLLMTurboQuantKVCacheCPU | CPU benchmark framing | CPU-audio and no-GPU regression lane | Low | `GFXATOM_KV_CPU_BENCH=1` |
| vllm_backend | Triton-style backend packaging | Cleaner deployment contracts | Low | `GFXATOM_VLLM_BACKEND_COMPAT=1` |
| wobble-quant-cache | Variance-aware allocation heuristic | Better per-dimension bit spend | Medium | `GFXATOM_KV_WOBBLE=1` |

## Fast Integration Sprint Plan

### Wave 1 (highest ROI, lowest risk)

1. `turbo-quant` Rust primitives (`src/turbo.rs`, `src/qjl.rs`, `src/kv.rs`)
2. `RateQuant` policy logic (`rate_distortion.py`, `mixed_precision.py`)
3. `norm-separated-quantization` guardrail logic (`compressors.py`)
4. `vLLMTurboQuantKVCacheCPU` benchmark adapter (`compute_metrics.py`)

Outcome:

- One canonical `kv_policy_family` selector with Rust-first primitives + safety guardrail + benchmark parity.

### Wave 2 (adaptive policies)

1. `delta-k-quantization` differential policy
2. `wobble-quant-cache` variance-aware allocator
3. `QAQ-KVCacheQuantization` quality-adaptive scorer

Outcome:

- Adaptive policy plugins behind strict compatibility contract.

### Wave 3 (tooling/deployment adapters)

1. `KVCache-Factory` comparative harness integration
2. `vllm_backend` deployment adapter docs/templates
3. `rust-llama.cpp` FFI adapter skeleton for gguf path
4. Optional `NautilusQuant` deterministic lane as experimental

Outcome:

- Operational comparability and cleaner runtime packaging.

### Wave 5 (control-plane routing metadata)

1. `llama-swap` capability profile for hot-swap, aliasing, TTL unload, groups, filters, and config reload

Outcome:

- Routing/lifecycle metadata becomes explicit in the engine runtime profile without introducing a gateway implementation.

### Wave 6 (model packing metadata)

1. `sardeenz` capability profile for multi-model packing, GPU distribution, kvcached memory sharing, move operations, and sleep/telemetry state

Outcome:

- GPU residency and model packing capabilities become explicit in the engine runtime profile without introducing a scheduler or dashboard.

### Wave 7 (compact runtime metadata)

1. `quant.cpp` capability profile for CPU-only runtime, on-demand download, Ollama-style CLI, OpenAI-compatible serving, progressive KV compression, and full-document mode

Outcome:

- Compact local-runtime capabilities become explicit in the engine runtime profile without introducing a new runtime stack.

### Wave 8 (quantized KV cache metadata)

1. `vllm` FP8 KV-cache capability profile for per-tensor/per-head scales, calibration, and non-fused quantized attention

Outcome:

- FP8 KV-cache detail becomes explicit in the engine runtime profile without adding a new attention backend.

### Wave 9 (decode-time KV compression policy)

1. `R-KV` redundancy-aware decode-time KV compression profile for observation tokens, redundancy windows, and budgeted top-k selection

Outcome:

- Decode-time KV compression becomes an explicit policy surface without changing the core runtime scheduler.

### Wave 10 (storage orchestration metadata)

1. `tair-kvcache` storage orchestration profile for distributed pooling, dynamic multilevel caching, metadata management, and capacity-aware KV matching

Outcome:

- Storage orchestration capability becomes explicit in the engine runtime profile without adding a new backend or scheduler path.

### Wave 11 (quantization pipeline metadata)

1. `llm-compressor` quantization pipeline profile for model-free PTQ, compressed-tensors output, and weight/activation/KV/attention transforms

Outcome:

- Quantization-pipeline capability becomes explicit in the engine runtime profile without introducing a separate deploy-time backend.

### Wave 12 (serving benchmark telemetry)

1. `rvllm` serving benchmark profile for Rust-native GPU serving, pure JAX TPU serving, cross-device benchmarking, and measurement telemetry

Outcome:

- Serving/benchmark capability becomes explicit in the engine runtime profile without pulling the TPU/GPU stack into gfxATOM.

### Wave 13 (rvLLM KV control-plane contracts)

1. `rvllm` KV transfer/prefetch/eviction/graph/telemetry contract imports for shared policy and validation layers

Outcome:

- rvLLM’s richer Rust KV contract shapes become reusable in gfxATOM without coupling the runtime to rvLLM-specific code.

### Wave 14 (rvLLM graph capture pool)

1. `rvllm` graph capture pool with metadata-layout hash gating and typed replay rejection

Outcome:

- The captured-graph replay contract becomes portable and reusable in gfxATOM without CUDA-specific graph code.

### Wave 15 (rvLLM radix snapshot)

1. `rvllm` radix snapshot contract for compact prefix-tree export and replay tooling

Outcome:

- Prefix-cache export becomes portable through a compact serializable radix snapshot type.

### Wave 16 (SMG dual-hash routing helpers)

1. `smg` dual-hash cache-aware routing helpers for token-block content hashes and chunked request hashing

Outcome:

- Prefix reuse and routing can share a deterministic content-hash primitive without depending on the donor repo.

### Wave 17 (rvLLM KV control-plane contracts)

1. `rvllm` KV transfer/prefetch/eviction/graph/telemetry contract imports for shared policy and validation layers

Outcome:

- rvLLM’s richer KV contract shapes became reusable in gfxATOM without coupling the runtime to rvLLM-specific code.

### Wave 18 (rvLLM graph capture pool)

1. `rvllm` graph capture pool with metadata-layout hash gating and typed replay rejection

Outcome:

- The captured-graph replay contract became portable and reusable in gfxATOM without CUDA-specific graph code.

### Wave 19 (rvLLM radix snapshot)

1. `rvllm` radix snapshot contract for compact prefix-tree export and replay tooling

Outcome:

- Prefix-cache export became portable through a compact serializable radix snapshot type.

### Wave 20 (SMG dual-hash routing helpers)

1. `smg` dual-hash cache-aware routing helpers for token-block content hashes and chunked request hashing

Outcome:

- Prefix reuse and routing can share a deterministic content-hash primitive without depending on the donor repo.

### Wave 21 (SMG prefix-match contract)

1. `smg` prefix-match result contract for tenant IDs, match counts, and hit-ratio bookkeeping

Outcome:

- Prefix-match routing bookkeeping now has a reusable contract surface with deterministic hit-ratio helpers.

### Wave 22 (quant.cpp delta-k KV delta compression)

1. `quant.cpp` delta-k policy lane for key-delta compression metadata

Outcome:

- Delta-k now has a dedicated policy profile surface and validation coverage without changing the runtime backend contract.

### Wave 23 (wobble-quant-cache variance-weighted KV allocation)

1. `wobble-quant-cache` variance-aware bit allocation lane for adaptive KV compression

Outcome:

- Wobble now has a dedicated variance-weighted policy profile and validation coverage without changing the runtime backend contract.

### Wave 24 (QAQ quality-adaptive KV quantization)

1. `QAQ-KVCacheQuantization` quality-targeted lane for KV compression decisions

Outcome:

- QAQ now has a dedicated quality-adaptive policy profile and validation coverage without changing the runtime backend contract.

### Wave 25 (Nautilus golden-ratio KV profile)

1. `NautilusQuant` deterministic golden-ratio lane for KV quantization geometry

Outcome:

- Nautilus now has a dedicated golden-ratio policy profile and validation coverage without changing the runtime backend contract.

### Wave 26 (ik_llama.cpp layer offloading metadata)

1. `ik_llama.cpp` layer-offloading capability lane for CPU/hybrid placement support

Outcome:

- Layer offloading is now explicit in the runtime profile contract without changing execution behavior.

### Wave 27 (sglang-1-bit-turbo autoquant observer snapshot)

1. `sglang-1-bit-turbo` observer-snapshot parity for autoquant policy validation

Outcome:

- Autoquant observer snapshots now round-trip through the Rust policy crate and validation harness.

### Wave 28 (gfxGRAPH graph bridge runtime profile)

1. `gfxGRAPH` runtime profile lane for graph shape bucketing, validation, conditional nodes, and nested capture

Outcome:

- gfxGRAPH graph-bridge capabilities are now explicit in the runtime profile contract with matching Python/Rust parity coverage.

### Wave 29 (vLLM family runtime profile)

1. `vllm` / `vllm-omni` / `vllm-ascend` runtime-family lane for multimodal serving, omni modality, and hardware plugin metadata

Outcome:

- The vLLM donor family now contributes a portable runtime capability seam without changing backend execution behavior.

### Wave 30 (vLLM family runtime profile extension)

1. `vllm` / `vllm-omni` / `vllm-ascend` runtime-family lane for additional adapter metadata and parity coverage

Outcome:

- The runtime profile contract now includes the vLLM family seam in both Python and Rust with focused validation coverage.

## Toggle/Configuration Contract

All extracted features are opt-in by default and must remain fail-closed.

- Global master switch: `GFXATOM_DONOR_FEATURES=0|1`
- Per-feature flags from table above.
- Arbitration keys:
  - `GFXATOM_KV_POLICY_FAMILY` = `ratequant|deltak|wobble|qaq|nautilus|baseline`
  - `GFXATOM_KV_POLICY_MODE` = `strict|fallback`

If a policy is unsupported on current engine/model shape:

- log explicit rejection reason
- fall back to baseline policy
- emit telemetry event `kv_policy_rejected`

## Consul/Kong Registration Plan (Governance-Compliant)

Reference:

- `/home/local/ai/docs/consul/consul-port-governance.md`

### Proposed service registration (new engine)

- Inference engine service name: `gfxatom-rust` (engine domain, `400xx`)
- Candidate port: `40013` (currently free in `consul.env`)
- API facade service name: `api-gfxatom` (`93xx`)
- Candidate port: `9314` (currently free in `consul.env`)

### Required pre-registration checks

1. `curl -s http://localhost:9500/v1/catalog/services | jq .`
2. `cat /home/local/ai/consul/consul.env`
3. `ss -ltnp | grep :40013`
4. `ss -ltnp | grep :9314`

### Registration workflow

1. Add JSON service entries under correct domain files in `/home/local/ai/consul/services/`:
   - engine route in mesh/engine mapping using `40013`
   - API route in `api` domain using `9314`
2. Regenerate env mappings:
   - `python3 scripts/generate_consul_env.py`
3. Bind runtime only through `${PORT_*}` variables (no hardcoded ports).
4. Configure Kong using dedicated vars (`PORT_KONG_PROXY`, `PORT_KONG_ADMIN`, `PORT_KONG_MANAGER`) and route `api-gfxatom` upstream.

### Required tags/metadata

- Domain tag: `inference` for engine, `api` for API facade
- Lifecycle tag: `always_on` or `on_demand` (explicitly chosen)
- `meta.project`: `ai` (or tighter module owner once finalized)

## Deliverables to Produce During Implementation

For each integrated donor feature:

1. One feature note file under `gfxATOM-Rust/docs/features/<feature>.md`:
   - source donor and implementation path
   - integration touchpoints
   - enabled/disabled behavior
   - fallback behavior
   - telemetry fields
2. One contract test proving:
   - enable path works
   - disable path bypasses cleanly
   - unsupported runtime rejects and falls back safely
3. One benchmark record:
   - throughput/latency
   - quality proxy (ppl or acceptance proxy)
   - memory/KV footprint

## Immediate Recommendation

Start with Wave 1 only, keep all features gated OFF by default, and enable per-feature canary flags in benchmark harness before any production default changes.
