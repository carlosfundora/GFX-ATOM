# Unified CLI Interface Specification: AMD TurboQuant Integration

**Date:** 2026-05-17  
**Scope:** Canonical CLI flags and API contracts for SGLang, vLLM, and llama.cpp  
**Goal:** Ensure consistent end-user experience across all inference engines  

---

## Overview

We are standardizing the CLI interface for TurboQuant/RotorQuant KV quantization support across three AMD-optimized inference engines. This document defines:

1. **Required flags** all engines must expose
2. **API contracts** for KV quantization backend configuration
3. **Environment variables** for tuning and debugging
4. **Error/warning messages** for user guidance
5. **Parity requirements** for feature completeness

---

## Core Feature Matrix

| Feature | SGLang | vLLM | llama.cpp | Status | Notes |
|---------|--------|------|-----------|--------|-------|
| `--kv-cache-dtype` flag | ✅ | ⏳ Add | ⏳ Add | Required | Selects quantization mode |
| TurboQuant modes (tq2, tq3, tq4) | ✅ Impl | ⏳ Add | ⏳ Add | Required | 2-4 bit KV |
| RotorQuant modes (rq3, rq4) | ✅ Impl | ⏳ Plan | ⏳ Plan | Optional | 3-4 bit rotational |
| FP8 modes (e4m3, e5m2) | ✅ Impl | ⏳ Plan | ⏳ Plan | Optional | Native FP8 support |
| INT8 mode | ✅ Impl | ⏳ Plan | ⏳ Plan | Optional | Signed 8-bit integer |
| Fallback chain | ✅ Impl | ⏳ Add | ⏳ Add | Required | Graceful degradation |
| Hardware detection | ✅ Impl | ⏳ Plan | ⏳ Plan | Required | Enforce gfx1030 only |
| Telemetry | ✅ Design | ⏳ Add | ⏳ Add | Required | Compression ratio logging |

---

## Section 1: CLI Flags (Primary Interface)

### 1.1 KV Cache Dtype Flag

**Flag Name:** `--kv-cache-dtype`

**Type:** Choice / Enum

**Valid Values:**
```
fp16               Native FP16 (baseline, always supported)
fp8_e4m3           FP8 E4M3 (native, if supported)
fp8_e5m2           FP8 E5M2 (native, if supported)
int8               Signed INT8 (native, if supported)
int4               Signed INT4 (native, if supported)
tq2                TurboQuant 2-bit (gfxATOM backend)
tq3                TurboQuant 3-bit (gfxATOM backend)
tq4                TurboQuant 4-bit (gfxATOM backend)
tq1                TurboQuant 1-bit (experimental, off by default)
tq8                TurboQuant 8-bit (reference, disabled)
rq3_planar         RotorQuant 3-bit planar (gfxATOM backend)
rq4_planar         RotorQuant 4-bit planar (gfxATOM backend)
rq3_iso            RotorQuant 3-bit isometric (gfxATOM backend)
rq4_iso            RotorQuant 4-bit isometric (gfxATOM backend)
```

**Default:** `fp16` (for compatibility)

**Recommended Default (gfx1030):** `tq2` (after validation in Phase 6)

**Example Usage:**
```bash
# SGLang
python -m sglang.launch_server --kv-cache-dtype tq2

# vLLM
python -m vllm.entrypoints.openai.api_server --kv-cache-dtype tq2

# llama.cpp
./main -m model.gguf --kv-cache-dtype tq2
```

### 1.2 Attention Backend Flag (Optional, AMD-only)

**Flag Name:** `--attention-backend` (SGLang) or `--gpu-backend` (vLLM/llama.cpp)

**Purpose:** Override automatic AMD backend selection

**Valid Values (AMD-only):**
```
aiter              Aiter attention kernel (AMD-optimized)
wave               Wave attention kernel (AMD-optimized)
triton             Triton fallback (CPU emulation, slow)
```

**Default:** Auto-detect (use best available)

**Example:**
```bash
python -m sglang.launch_server --kv-cache-dtype tq2 --attention-backend wave
```

---

## Section 2: Environment Variables (Advanced Tuning)

### 2.1 TurboQuant Tuning

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `SGLANG_KV_CACHE_TURBOQUANT_ROPE` | bool | false | Enable RoPE in TurboQuant pipeline |
| `TURBOQUANT_QJL` | bool | false | Enable QJL projection in TurboQuant |
| `TURBOQUANT_SEED` | int | 42 | Deterministic seed for quantization |
| `TURBOQUANT_POLAR_ORDER` | int | 8 | Polar coordinate transform order |
| `AUTOQUANT_POLICY` | string | "balanced" | AutoQuant policy: "aggressive", "balanced", "conservative" |
| `KV_CACHE_DEBUG` | bool | false | Verbose telemetry logging |

**Example:**
```bash
export SGLANG_KV_CACHE_TURBOQUANT_ROPE=true
export TURBOQUANT_SEED=2024
python -m sglang.launch_server --kv-cache-dtype tq2
```

### 2.2 AMD Hardware Detection

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `ROCM_DEVICE` | int | 0 | GPU device ID (0-based) |
| `ENFORCE_GFX1030` | bool | false | Reject if GPU is not gfx1030 |
| `AMD_HARDWARE_CHECK_VERBOSE` | bool | false | Print detected GPU architecture |

**Example:**
```bash
export ENFORCE_GFX1030=true
python -m sglang.launch_server --kv-cache-dtype tq2
```

---

## Section 3: Configuration File Support

**File:** `kv_quantization_config.json` (optional, per-engine)

**Purpose:** Persistent configuration without CLI flags

**Schema (unified across all engines):**
```json
{
  "kv_cache_dtype": "tq2",
  "attention_backend": "auto",
  "enable_prefix_reuse": true,
  "max_tokens": 32000,
  "page_size": 16,
  "compression_policy": "aggressive",
  "fallback_chain": ["tq2", "fp8_e4m3", "fp16"],
  "telemetry": {
    "log_compression_ratio": true,
    "log_kv_allocation": true,
    "log_backend_selection": true
  },
  "experimental": {
    "enable_1bit_mode": false,
    "enable_rotor_quant": false
  }
}
```

**Usage:**
```bash
# SGLang
python -m sglang.launch_server --config-file kv_quantization_config.json

# vLLM
python -m vllm.entrypoints.openai.api_server --config-file kv_quantization_config.json
```

---

## Section 4: API Contracts (Internal)

### 4.1 Backend Factory Interface

**Rust Signature (gfxATOM):**
```rust
pub trait KVQuantizationBackend {
    fn encode(&self, kv_data: &Tensor) -> Result<CompressedKV>;
    fn decode(&self, compressed: &CompressedKV) -> Result<Tensor>;
    fn estimate_inner_product(&self, q: &Tensor, compressed: &CompressedKV) -> Result<Tensor>;
    fn get_compression_ratio(&self) -> f32;
    fn supports_hardware(&self, hw: &HardwareProfile) -> bool;
}
```

**Python Wrapper (for SGLang/vLLM):**
```python
class KVQuantizationBackend(ABC):
    @abstractmethod
    def encode(self, kv_data: torch.Tensor) -> CompressedKV:
        pass

    @abstractmethod
    def decode(self, compressed: CompressedKV) -> torch.Tensor:
        pass

    @abstractmethod
    def estimate_inner_product(self, q: torch.Tensor, compressed: CompressedKV) -> torch.Tensor:
        pass

    @abstractmethod
    def get_compression_ratio(self) -> float:
        pass

    @abstractmethod
    def supports_hardware(self, hw_profile: Dict) -> bool:
        pass
```

### 4.2 Backend Selection Logic

**Flow (all engines):**
```
1. Parse CLI flag: --kv-cache-dtype tq2
2. Detect hardware: gfx1030 AMD GPU
3. Validate: Is (tq2, gfx1030) supported?
   YES → Continue
   NO  → Log error + suggest fallback
4. Load backend: new TurboQuantizer(config)
5. Test: encode/decode smoke test
6. Enable: Use for all subsequent KV compression
7. Fallback: If runtime error, use fallback_chain
```

### 4.3 Fallback Chain Semantics

**Default Fallback Chain (all engines):**
```
tq2 → tq3 → tq4 → fp8_e4m3 → fp16
```

**Behavior:**
- If primary backend fails to initialize, try next in chain
- Log warning for each fallback step
- Preserve user's choice in error logs for debugging
- Never silently drop to FP16 without warning

---

## Section 5: Telemetry Output

### 5.1 Metrics to Log (Standard Across All Engines)

**Format:** JSON to stdout (one record per batch or per N seconds)

```json
{
  "timestamp": "2026-05-17T12:34:56Z",
  "backend": "TurboQuant_2bit",
  "metrics": {
    "compression_ratio": 0.25,
    "kv_allocated_bytes": 2097152,
    "kv_capacity_bytes": 8388608,
    "kv_usage_percent": 25.0,
    "prefill_tokens_per_second": 450.3,
    "decode_tokens_per_second": 28.5,
    "kernel_latency_us": 1250,
    "memory_bandwidth_gbps": 450.0,
    "gpu_free_mb": 4096,
    "gpu_used_mb": 8192,
    "prefix_reuse_ratio": 0.35,
    "fallback_events": 0
  }
}
```

### 5.2 Log Levels

| Level | Condition | Action |
|-------|-----------|--------|
| INFO | Normal operation | Log backend name, compression ratio |
| WARN | Fallback triggered | Log why fallback occurred, which chain step |
| WARN | Hardware unsupported | Log GPU arch, required arch, suggested workaround |
| ERROR | Backend initialization failed | Log config, error message, fallback chain |
| ERROR | User flag invalid | Log flag, valid choices, example usage |

**Example INFO:**
```
[INFO] KV quantization backend: TurboQuantizer (2-bit mode)
[INFO] Expected compression ratio: 0.25 (16x reduction vs FP16)
[INFO] Target hardware: AMD gfx1030 (verified)
```

**Example WARN:**
```
[WARN] TurboQuant 2-bit backend not available; falling back to TurboQuant 3-bit
[WARN] Expected compression ratio: 0.375 (vs 0.25 requested)
```

**Example ERROR:**
```
[ERROR] Invalid --kv-cache-dtype: "tq5" not supported
[ERROR] Valid choices: fp16, fp8_e4m3, fp8_e5m2, int8, int4, tq1, tq2, tq3, tq4, tq8, rq3_planar, rq4_planar, rq3_iso, rq4_iso
[ERROR] Defaulting to fp16 (no quantization)
```

---

## Section 6: Platform-Specific Behavior

### 6.1 AMD (gfx1030) Required Behavior

| Item | Requirement | Rationale |
|------|-------------|-----------|
| Default dtype | Suggest `tq2` in help text | Best compression/quality tradeoff |
| Attention backend | Default to `wave` or `aiter` | AMD-optimized kernels |
| Page size | Default to 16 tokens | Memory alignment on RDNA2 |
| Max context | Adaptive based on VRAM | 12GB typical on RX 6700 XT |
| Feature gates | Experimental modes off by default | Stability-first approach |

### 6.2 Unsupported Platform Detection

**If GPU is NOT gfx1030 or ROCm not available:**

```
[ERROR] TurboQuant KV quantization requires AMD ROCm 7.2+
[ERROR] Detected: CUDA 12.0 (NVIDIA GPU)
[ERROR] Please use:
  - AMD GPU: Radeon RX 6700 XT or higher
  - Or set --kv-cache-dtype fp16 to disable quantization
[ERROR] Exiting.
```

---

## Section 7: Rollout Phases

### Phase 1: SGLang (Week 1-2) ✅
- ✅ Flags: --kv-cache-dtype tq2/tq3/tq4
- ✅ Backends: TurboQuantizer + Triton fallback
- ✅ Telemetry: Compression ratio + allocation metrics
- ✅ Hardware checks: gfx1030 detection

### Phase 2: vLLM (Week 3, conditional)
- ⏳ Add QuantConfig subclass for TurboQuant
- ⏳ Wire flags: --kv-cache-dtype tq2/tq3/tq4
- ⏳ Match SGLang telemetry schema
- ⏳ Validate AMD ROCm QuantConfig path

### Phase 3: llama.cpp (Week 4+, optional)
- ⏳ Investigate GGML quantization hooks
- ⏳ Add CLI support for tq modes
- ⏳ Wire to AMD HIP backend
- ⏳ Provide binary + source diffs

---

## Section 8: Validation Checklist

Before marking an engine as "parity complete":

- [ ] `--kv-cache-dtype` flag recognized and validated
- [ ] TurboQuant modes (tq2, tq3, tq4) instantiate correctly
- [ ] Fallback chain works (e.g., tq2 → fp16)
- [ ] AMD hardware detected (gfx1030 confirmed)
- [ ] Telemetry logs compression ratio + allocation metrics
- [ ] Error messages follow schema (JSON to stdout)
- [ ] Existing tests pass (no regressions)
- [ ] E2E test: 10 token generation with tq2 (Qwen-1.5B)
- [ ] Documentation updated (flags, examples, troubleshooting)
- [ ] Help text includes TurboQuant options (`--help | grep kv-cache`)

---

## Section 9: Known Limitations & Escape Hatches

| Limitation | Workaround | Timeline |
|------------|-----------|----------|
| tq1 (1-bit) not implemented | Use tq2 (2-bit) instead | Phase 5.3 for 1-bit |
| RotorQuant kernels pending | Use TurboQuant modes only | Phase 5.2+ for rotor |
| No dynamic quantization | Quantization is static at server startup | Phase 6+ for dynamic |
| Single gfx1030 support | No multi-GPU or mixed architectures yet | Phase 7+ for multi-GPU |

---

## Section 10: Examples

### SGLang Example (Primary)

```bash
# Start server with TurboQuant 2-bit KV
python -m sglang.launch_server \
  --model-path Qwen/Qwen-7B \
  --kv-cache-dtype tq2 \
  --port 8000

# Query the server
curl http://localhost:8000/v1/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 100}'
```

### vLLM Example (SECONDARY, when ready)

```bash
# Start vLLM API with TurboQuant
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen-7B \
  --kv-cache-dtype tq2 \
  --port 8001
```

### llama.cpp Example (TERTIARY, when ready)

```bash
# Run llama.cpp with TurboQuant
./main \
  --model model.gguf \
  --kv-cache-dtype tq2 \
  --prompt "Hello" \
  --n-predict 100
```

---

## Section 11: Related Documentation

- **AMD_ENGINE_SUPPORT_AUDIT.md** — Platform-specific AMD support details
- **CROSS_ENGINE_PARITY_ANALYSIS.md** — Feature-by-feature parity matrix
- **PHASE4_SGLANG_CONFIG_AUDIT.md** — SGLang configuration layer details
- **PHASE4_KV_LAYOUT_AUDIT.md** — KV memory layout compatibility

---

## Appendix: Help Text Templates

### SGLang Help Text

```
--kv-cache-dtype {fp16,fp8_e4m3,fp8_e5m2,int8,int4,tq1,tq2,tq3,tq4,tq8,rq3_planar,rq4_planar,rq3_iso,rq4_iso}
    KV cache quantization format.
    
    Supported modes:
      fp16              Native FP16 (baseline, always available)
      fp8_e4m3          FP8 with 4-bit exponent, 3-bit mantissa
      fp8_e5m2          FP8 with 5-bit exponent, 2-bit mantissa
      int8              Signed 8-bit integer
      int4              Signed 4-bit integer
      tq2               TurboQuant 2-bit (8x compression, gfx1030 AMD)
      tq3               TurboQuant 3-bit (5.3x compression, gfx1030 AMD)
      tq4               TurboQuant 4-bit (4x compression, gfx1030 AMD)
      tq1               TurboQuant 1-bit (16x compression, EXPERIMENTAL)
      tq8               TurboQuant 8-bit (reference, disabled)
      rq3_planar        RotorQuant 3-bit planar (gfx1030 AMD)
      rq4_planar        RotorQuant 4-bit planar (gfx1030 AMD)
      rq3_iso           RotorQuant 3-bit isometric (gfx1030 AMD)
      rq4_iso           RotorQuant 4-bit isometric (gfx1030 AMD)
    
    Fallback chain: tq2 → tq3 → tq4 → fp8_e4m3 → fp16
    
    For AMD gfx1030 GPU, recommend: --kv-cache-dtype tq2
    
    Default: fp16
    
    Example:
      python -m sglang.launch_server --kv-cache-dtype tq2
```

---

**Status:** READY FOR IMPLEMENTATION  
**Last Updated:** 2026-05-17  
**Next Review:** After Phase 4.4 implementation
