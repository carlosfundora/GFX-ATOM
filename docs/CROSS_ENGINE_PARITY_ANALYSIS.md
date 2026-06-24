# Cross-Engine Parity Analysis: TurboQuant Integration

**Date:** 2026-05-17  
**Scope:** Comparing TurboQuant support across SGLang, vLLM, and llama.cpp  
**Goal:** Ensure unified KV quantization interface across all inference engines

---

## Executive Summary

We are integrating TurboQuant (2-8 bit KV quantization) into three major inference engines to provide a unified, portable compression strategy for gfx1030 AMD GPUs. This document tracks parity across all three engines and audio integration points.

### Key Status

| Engine | TurboQuant Support | Status | Notes |
|--------|-------------------|--------|-------|
| **SGLang-1-bit-turbo** | tq2, tq3, tq4 | ✅ Native | Flags in server_args.py (L4087-4089) |
| **vLLM-1-bit-turbo** | ? | 📋 AUDIT | Need to check quant_config.py |
| **llama.cpp-1-bit-turbo** | ? | 📋 AUDIT | Need to check quantization layer |
| **llama.cpp-audio-max** | ❌ None yet | 📋 PLAN | Compiled binary, no source mods |
| **lfm2.5-audio-fastapi** | ❌ None yet | 📋 PLAN | Python FastAPI, can add adapter |

---

## Engine Breakdown

### 1. SGLang-1-bit-turbo ✅

**Location:** `donors/sglang-1-bit-turbo/`

**Quantization Layer:**
```
python/sglang/srt/server_args.py:4087-4089
  --kv-cache-dtype: [fp16, fp8_e4m3, fp8_e5m2, int8, int4, tq2, tq3, tq4]
  
python/sglang/srt/mem_cache/memory_pool.py
  ReqToTokenPool → Token-to-KV mapping
```

**TurboQuant Support:**
- ✅ Flags: tq2, tq3, tq4 present
- ✅ Env vars: SGLANG_KV_CACHE_TURBOQUANT_ROPE, TURBOQUANT_QJL
- ✅ Help text: Documented (lines 4097-4105)
- ⚠️ Missing: tq1 (experimental), tq8 (reference)
- ⚠️ Status: Flags present but no encode/decode impl visible

**Integration Path:**
```
server_args.py (flag parsing)
  ↓
model_executor.py (backend selection)
  ↓
memory_pool.py (KV storage)
  ↓
attention_layers.py (compression hooks)
```

**Next Steps:**
1. Add tq1 and tq8 to choices
2. Trace encode/decode implementation path
3. Wire gfxATOM TurboQuantizer

### 2. vLLM-1-bit-turbo 📋

**Location:** `donors/vllm-1-bit-turbo/`

**Quantization Layer:**
```
vllm/model_executor/layers/quantization/quant_config.py
  QuantConfig (base class for all quant modes)
  
vllm/model_executor/layers/quantization/
  ├─ compressed_tensors_quant/
  ├─ fbgemm_fp8/
  ├─ awq/
  ├─ gptq/
  ├─ marlin/
  └─ ???_turbo/  [TO BE DETERMINED]
```

**TurboQuant Support:**
- ❓ Flags: NEED TO CHECK
- ❓ Backend: NEED TO CHECK
- ❓ Kernel path: NEED TO CHECK
- ⚠️ vLLM has many quant backends; need to find where TurboQuant fits

**Integration Path:**
```
vllm.entrypoints.llm (CLI args)
  ↓
vllm.model_executor (QuantConfig)
  ↓
vllm.worker (KV pool)
  ↓
vllm.attention (kernel selection)
```

**TODO:** 
- [ ] Search for "turbo" or "tq" in vllm-1-bit-turbo
- [ ] Find QuantConfig subclass for TurboQuant
- [ ] Map to attention kernel layer
- [ ] Identify missing tq1/tq8 support

### 3. llama.cpp-1-bit-turbo 📋

**Location:** `donors/llama.cpp-1-bit-turbo/`

**Quantization Layer:**
```
common/common.cpp (quantization enums)
common/quantize.cpp (quantization logic)
ggml/src/ggml.c (tensor ops)
```

**TurboQuant Support:**
- ❓ Enum: NEED TO CHECK ggml_type
- ❓ Backend: NEED TO CHECK quantize.cpp
- ❓ HIP/GPU path: NEED TO CHECK
- ⚠️ llama.cpp uses GGML abstraction; may not have KV quant at all

**Integration Path:**
```
main.cpp (CLI: --kv-cache-dtype)
  ↓
llama.cpp (model context setup)
  ↓
ggml/src/ggml.c (tensor graph)
  ↓
GGML GPU backend (HIP/CUDA/Metal)
```

**TODO:**
- [ ] Check if KV quantization supported in llama.cpp at all
- [ ] Search for quant enums in ggml_type
- [ ] Check GPU dispatch in ggml_backend
- [ ] Evaluate feasibility vs. recompile cost

---

## Audio Integration Points

### 4. llama.cpp-audio-max 🎙️

**Location:** `donors/llama.cpp-audio-max/` (symlink to /home/local/ai/engines/)

**Architecture:**
- Compiled binary (no source modifications)
- Built for audio processing with llama.cpp backend
- Uses external Whisper/TTS services

**Capabilities:**
- Speech recognition (via external service)
- Audio streaming input
- Text-to-speech output (external)

**TurboQuant Integration:**
- ⚠️ Binary rebuild required to enable TurboQuant
- Option 1: Rebuild llama.cpp with TurboQuant flags
- Option 2: Use llama-cpp-python wrapper with dynamic config

**TODO:**
- [ ] Check if rebuild needed or Python wrapper sufficient
- [ ] Add --kv-cache-dtype tq2 to command line
- [ ] Verify audio latency impact of TurboQuant

### 5. lfm2.5-audio-fastapi 🎙️

**Location:** `donors/lfm2.5-audio-fastapi/`

**Architecture:**
- Python FastAPI server
- Configurable inference backend (can use SGLang/vLLM/llama.cpp)
- Real-time audio streaming

**Current Stack:**
```python
server.py:
  - FastAPI endpoints for audio in/out
  - Pluggable inference backends
  - Event streaming
```

**TurboQuant Integration:**
- ✅ Can pass flags to SGLang backend
- ✅ Can pass flags to vLLM backend
- ⚠️ Need to add tq_mode parameter to audio API

**TODO:**
- [ ] Add --kv-cache-dtype param to audio API
- [ ] Wire to underlying engine (SGLang/vLLM/llama.cpp)
- [ ] Measure latency impact on streaming

---

## Parity Checklist

### Configuration Flags

| Engine | fp16 | fp8 | int8 | int4 | tq1 | tq2 | tq3 | tq4 | tq8 |
|--------|------|-----|------|------|-----|-----|-----|-----|-----|
| SGLang | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| vLLM | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| llama.cpp | ? | ? | ? | ? | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Target** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### KV Memory Management

| Engine | Paged KV | Prefix Reuse | Radix Cache | GPU Offload |
|--------|----------|--------------|-------------|------------|
| SGLang | ✅ | ✅ | ✅ | ✅ |
| vLLM | ✅ | ⚠️ (GH #3921) | ❌ | ✅ |
| llama.cpp | ❌ | ❌ | ❌ | ⚠️ (GGML) |

### Audio Integration

| Service | SGLang | vLLM | llama.cpp |
|---------|--------|------|-----------|
| lfm2.5-audio-fastapi | ✅ | ⚠️ | ⚠️ |
| llama.cpp-audio-max | ❌ | ❌ | ✅ |
| Pipecat voice | ❌ | ❌ | ❌ |

---

## Recommended Integration Order

### Phase 1: Core Engines (Weeks 1-2)
1. **SGLang** (DONE): Add tq1/tq8, wire TurboQuantizer
2. **vLLM**: Audit quant_config, add TurboQuantizer backend
3. **llama.cpp**: Determine if KV quant feasible, evaluate cost

### Phase 2: Audio Integration (Weeks 2-3)
1. **lfm2.5-audio-fastapi**: Add --kv-cache-dtype parameter
2. **llama.cpp-audio-max**: Rebuild with TurboQuant (if feasible)
3. **Pipecat voice**: Consider future integration

### Phase 3: Production Hardening (Weeks 3-4)
1. Cross-engine latency benchmarks
2. Accuracy floor validation
3. Feature parity documentation

---

## vLLM Audit TODO

**Search Paths:**
```
donors/vllm-1-bit-turbo/
  ├─ vllm/model_executor/layers/quantization/
  ├─ vllm/worker.py (KV pool instantiation)
  ├─ vllm/attention/
  └─ vllm/entrypoints/llm.py (CLI args)
```

**Key Questions:**
1. Are TurboQuant flags present in QuantConfig?
2. How does vLLM route quant dtype → kernel?
3. Can vLLM KV pool be extended for TurboQuant?
4. What's the equivalent of SGLang's radix cache?

**Success Criteria:**
- [ ] vLLM can start with `--kv-cache-dtype tq2`
- [ ] vLLM KV pool stores compressed tokens
- [ ] vLLM attention uses TurboQuantizer for IP estimation

---

## llama.cpp Audit TODO

**Search Paths:**
```
donors/llama.cpp-1-bit-turbo/
  ├─ common/common.cpp (ggml_type enum)
  ├─ common/quantize.cpp (quantize logic)
  ├─ ggml/src/ggml.c (tensor ops)
  └─ src/llama.cpp (KV cache layer)
```

**Key Questions:**
1. Does llama.cpp support KV cache quantization at all?
2. Is GGML flexible enough for TurboQuant?
3. How hard is it to add a new GGML type?
4. What's the rebuild cost vs. wrapper approach?

**Success Criteria (or Decision to Defer):**
- [ ] Determine if KV quant is feasible in GGML
- [ ] If yes: Add tq2-tq4 enums to ggml_type
- [ ] If no: Document limitation and use wrapper approach

---

## Unified Interface Specification

**All engines should support this:**

```bash
# CLI invocation
sglang server --model qwen-1.5b --kv-cache-dtype tq2
vllm serve --model qwen-1.5b --kv-cache-dtype tq2
llama-cpp-python --model ./qwen-1.5b.gguf --kv-cache-dtype tq2

# Audio API
curl -X POST http://localhost:9300/audio/chat \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello",
    "model": "qwen-1.5b",
    "kv_cache_dtype": "tq2",
    "streaming": true
  }'
```

**Adapter Bridge (gfxATOM side):**
```
engine_config.kv_cache_dtype (string: "tq2")
  ↓ (via CodecAdapterRegistry)
KvCodec.tq2
  ↓ (via SGLangTurboQuantAdapter / vLLMTurboQuantAdapter / etc)
TurboQuantizer.encode_kv()
```

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| vLLM quant layer incompatibility | HIGH | Early audit; may need to fork quantizer |
| llama.cpp GGML rigidity | HIGH | Determine feasibility early; plan fallback |
| Audio latency impact | MEDIUM | Benchmark streaming turnaround time |
| Cross-engine accuracy variance | MEDIUM | Property-based tests; accuracy floors |
| Rebuild/maintenance burden | MEDIUM | Prioritize SGLang/vLLM; defer llama.cpp if needed |

---

## Next Steps

1. **Immediate (Today):**
   - [ ] Audit vLLM-1-bit-turbo/python/sglang/srt/model_executor/layers/quantization/
   - [ ] Search for "turbo" or "tq" markers in vLLM fork
   - [ ] Check llama.cpp-1-bit-turbo for GGML quant extensions

2. **Phase 4.3 Integration:**
   - [ ] Wire SGLang TurboQuantizer (planned)
   - [ ] Add vLLM TurboQuantizer adapter (new)
   - [ ] Document llama.cpp feasibility (new)

3. **Audio Integration (Phase 4.4+):**
   - [ ] Add --kv-cache-dtype to lfm2.5-audio-fastapi
   - [ ] Rebuild llama.cpp-audio-max with TurboQuant (if feasible)

---

## Owner & Approvals

- **Owner:** ROCmancer (Wave 33B integration)
- **Last Updated:** 2026-05-17T03:17 UTC
- **Status:** DRAFT (awaiting vLLM/llama.cpp audit)

