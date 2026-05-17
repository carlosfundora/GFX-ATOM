# AMD-Only Engine Support Audit: TurboQuant Integration

**Date:** 2026-05-17  
**Scope:** AMD/ROCm support in SGLang, vLLM, and llama.cpp forks  
**Goal:** Assess feasibility of AMD-only TurboQuant integration

---

## Executive Summary

We are integrating TurboQuant specifically for **AMD gfx1030 GPUs**. This document audits AMD/ROCm support in the three major inference engines and identifies which are viable candidates for AMD-optimized TurboQuant integration.

### Key Finding: llama.cpp is AMD-native, vLLM has AMD support, SGLang is AMD-ready

| Engine | AMD Support | Status | Priority |
|--------|-------------|--------|----------|
| **SGLang-1-bit-turbo** | ✅ Full | Native AMD backend (HWBackend.AMD) | 🔴 PRIMARY |
| **vLLM-1-bit-turbo** | ✅ Partial | current_platform.is_rocm() checks present | 🟡 SECONDARY |
| **llama.cpp-1-bit-turbo** | ✅ Full | Dedicated ggml-hip backend, ROCm Dockerfile | 🟢 TERTIARY |

---

## Engine-by-Engine Analysis

### 1. SGLang-1-bit-turbo ✅ PRIMARY

**AMD Support Status:** ✅ Full Native Support

**Evidence:**
```python
# test/run_suite.py (line: "amd": HWBackend.AMD)
HWBackend enum includes AMD as first-class backend

# test/registered/sampling/test_pytorch_sampling_backend.py
register_amd_ci(est_time=66, suite="stage-b-test-1-gpu-small-amd")
Dedicated AMD CI test suite
```

**Architecture:**
- Native AMD backend selection in hardware abstraction layer
- Dedicated AMD test suite with AMD CI registration
- Likely uses PyTorch ROCm backend + Aiter/Wave attention kernels

**TurboQuant Integration Path:**
```
SGLang CLI: --kv-cache-dtype tq2 (already present, lines 4087-4089)
  ↓ (via HWBackend.AMD + server_args)
model_executor → attention layer selection
  ↓
gfxATOM TurboQuantizer (encode/decode/IP estimation)
```

**Status:** ✅ READY (Phase 4.3 work in progress)

**AMD-Specific Optimizations:**
- Attention backend choices: Already includes `aiter` and `wave` (AMD-optimized)
- KV cache layout: Already supports paged KV (AMD-friendly)
- GPU memory: Already has offload policies (AMD VRAM-aware)

### 2. vLLM-1-bit-turbo ✅ SECONDARY

**AMD Support Status:** ✅ Partial (Platform checks present)

**Evidence:**
```cpp
// benchmarks/kernels/benchmark_paged_attention.py
if current_platform.is_rocm():

// benchmarks/kernels/benchmark_moe_defaults.py
"num_stages": 3 if not current_platform.is_rocm() else 2

// benchmarks/kernels/benchmark_w8a8_block_fp8.py
assert current_platform.is_cuda() or current_platform.is_rocm()
```

**Architecture:**
- Uses `current_platform.is_rocm()` abstraction for GPU detection
- vLLM has many quantization backends (awq, gptq, marlin, etc.)
- Appears to have platform-aware configuration tuning

**TurboQuant Integration Path:**
```
vLLM CLI: --kv-cache-dtype tq2 (need to add)
  ↓ (via QuantConfig subclass)
vllm/model_executor/layers/quantization/turboquant/
  ↓ (TurboQuantConfig extends QuantizationConfig)
gfxATOM TurboQuantizer adapter
```

**Status:** 🟡 CONDITIONAL
- Requires adding TurboQuantConfig backend
- Platform-aware code suggests AMD support is feasible
- May need to check how QuantConfig routes to kernels

**AMD-Specific Questions:**
- [ ] Does vLLM have AMD-optimized paged attention kernel?
- [ ] How does QuantConfig → kernel dispatch work?
- [ ] Are there AMD examples in the fork?

### 3. llama.cpp-1-bit-turbo ✅ TERTIARY

**AMD Support Status:** ✅ Full Native (dedicated HIP backend)

**Evidence:**
```cpp
// Dedicated AMD infrastructure
.devops/rocm.Dockerfile          (ROCm build environment)
.github/workflows/hip-quality-check.yml (AMD CI pipeline)
.github/workflows/amd-local-ci.yml      (AMD-specific testing)
tests/test-rocm-hardening.cpp   (AMD hardening tests)
.jules/journals/rocmancer.md    (AMD optimization notes)

// Backend implementation
ggml/src/ggml-hip/              (HIP backend for AMD)
ggml/src/ggml-cuda/vendors/hip.h (HIP compatibility layer)
```

**Architecture:**
- llama.cpp uses GGML (Graph-optimized Tensor Machine Learning)
- GGML has native HIP backend for AMD GPUs
- Dedicated ROCm Docker setup and CI pipeline
- AMD hardening tests indicate serious AMD support commitment

**TurboQuant Integration Path:**

⚠️ **COMPLEX: Requires GGML type extension**

```
llama.cpp CLI: --kv-cache-dtype tq2 (need to add)
  ↓ (parse to GGML enum)
ggml/src/ggml.c: Add GGML_TYPE_TQ2, etc.
  ↓ (via ggml-hip backend dispatch)
HIP kernel for TurboQuantizer
  ↓ (gfxATOM TurboQuantizer or custom HIP impl)
```

**Status:** 🟢 VIABLE (but more complex than SGLang/vLLM)

**Complexity Assessment:**
- **Easy:** Add CLI flag to main.cpp
- **Medium:** Extend GGML type enum (requires careful serialization)
- **Hard:** Implement HIP kernels or wire to gfxATOM from C++

**AMD-Specific Advantages:**
- ✅ Dedicated HIP backend (not CUDA-first)
- ✅ AMD testing infrastructure already in place
- ✅ ROCm hardening tests indicate AMD-native quality
- ⚠️ C++ integration more complex than Python

---

## Decision: AMD-Only Prioritization

### Tier 1: SGLang (IMMEDIATE)
- ✅ Already has TurboQuant flags
- ✅ Native AMD support (HWBackend.AMD)
- ✅ Python integration (easy)
- **Action:** Phase 4.3 backend factory wiring (in progress)

### Tier 2: vLLM (CONDITIONAL)
- ✅ Has current_platform.is_rocm() abstraction
- ✅ Many quantization backends to study
- ⚠️ Need to audit QuantConfig routing
- **Action:** Phase 4.2b audit; proceed if QuantConfig is extensible

### Tier 3: llama.cpp (DEFERRED)
- ✅ Full AMD support with dedicated HIP backend
- ⚠️ Requires GGML type extension (complex)
- ⚠️ C++ integration overhead
- **Action:** Defer to Phase 6+ if Phase 4+5 complete; consider if ROCm HIP kernels needed

---

## AMD-Focused Integration Strategy

### For SGLang (PRIMARY - Week 1-2)
1. ✅ Add tq1/tq8 to server_args.py choices
2. Wire gfxATOM TurboQuantizer into model_executor
3. Add encode/decode hooks in attention layers
4. Test with `--hw-backend amd` flag (if available)
5. Benchmark on gfx1030 hardware

### For vLLM (SECONDARY - Week 2-3, if feasible)
1. Audit QuantConfig extensibility for TurboQuantConfig
2. If feasible: Create vLLMTurboQuantAdapter
3. If not feasible: Document blocker; defer to Phase 6+
4. Benchmark if available

### For llama.cpp (TERTIARY - Defer unless critical)
1. Document GGML extension requirements
2. Prototype in Phase 6 if additional AMD inference needed
3. Consider if custom HIP kernels required for optimal gfx1030 perf

---

## Known AMD-Specific Constraints

| Engine | Constraint | Workaround |
|--------|-----------|-----------|
| SGLang | Aiter/Wave attention kernel availability | Already supported (check availability) |
| vLLM | HIP attention kernel selection | May default to CUDA; need platform override |
| llama.cpp | GGML serialization for new types | Requires careful binary format extension |

---

## Risk Assessment (AMD-Focused)

| Risk | Severity | Mitigation |
|------|----------|-----------|
| vLLM QuantConfig is CUDA-first | MEDIUM | Early audit (4.2b.1); defer if not AMD-friendly |
| llama.cpp GGML is complex | MEDIUM | Defer to Phase 6; focus on SGLang first |
| AMD attention kernel gaps | LOW | SGLang already has aiter/wave; vLLM can fallback |
| TurboQuant HIP kernel missing | MEDIUM | Can use CPU implementation via gfxATOM Rust bridge |

---

## Success Criteria (AMD-Only)

### Before Phase 5 (Algorithm Implementation):
- [ ] SGLang server starts with `--kv-cache-dtype tq2` (stubs acceptable)
- [ ] SGLang can detect gfx1030 via ROCm
- [ ] gfxATOM TurboQuantizer factory wired to SGLang backend
- [ ] vLLM audit complete (proceed or defer decision made)
- [ ] llama.cpp audit complete (defer decision made)

### Before GPU Benchmarking (Phase 6):
- [ ] SGLang encode/decode work on gfx1030
- [ ] Compression ratio ≥ 8x for tq2 mode
- [ ] Latency impact < 5% on small models
- [ ] Accuracy floor verified (< 10% for tq2)

### Optional (Phase 6+):
- [ ] vLLM support (if Phase 4.2b audit indicates feasibility)
- [ ] llama.cpp support (if additional inference engines needed)

---

## Deferred: CUDA-Only Engines

We are **not** integrating TurboQuant into:
- PyTorch bare (no inference server)
- Vino (Intel focus)
- MLC LLM (too specialized)
- Any CUDA-first engines

Focus remains exclusively on AMD/ROCm paths.

---

## Owner & Approvals

- **Owner:** ROCmancer (Wave 33B integration)
- **Scope:** AMD gfx1030 optimization only
- **Last Updated:** 2026-05-17T03:17 UTC
- **Status:** ACTIVE AUDIT (Phase 4.2b)

