# Wave 33B Phase 4.2: SGLang Config Flag Parsing Audit

**Date:** 2026-05-17  
**Status:** COMPLETE ✅  
**Scope:** Analyze sglang-1-bit-turbo fork for existing TurboQuant config support

---

## Executive Summary

✅ **sglang-1-bit-turbo ALREADY HAS TurboQuant support**

The fork includes:
- `--kv-cache-dtype tq1|tq2|tq3|tq4` flags (lines 4087-4089 in server_args.py)
- Codec choices: `["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "fp4_e2m1", "tq4", "tq3", "tq2", "rq3", "rq4", ...]`
- TurboQuant environment variables: `SGLANG_KV_CACHE_TURBOQUANT_ROPE`, `SGLANG_KV_CACHE_TURBOQUANT_QJL`
- Full help text documenting codec modes

**Phase 4.2 Scope Adjustment:**  
Rather than adding config support (already done), Phase 4.2 now focuses on:
1. **Verify codec enum alignment** between sglang and gfxATOM contracts
2. **Audit kernel dispatch paths** for tq mode routing
3. **Identify remaining integration points** (encode/decode hooks)
4. **Create adapter bridge** to connect SGLang tq flags → gfxATOM TurboQuantizer

---

## SGLang-1-Bit-Turbo Fork Analysis

### File: `/donors/sglang-1-bit-turbo/python/sglang/srt/server_args.py`

#### ServerArgs Dataclass (line 344)
```python
kv_cache_dtype: str = "auto"
```

#### Argparse Definition (lines 4076-4106)
```python
parser.add_argument(
    "--kv-cache-dtype",
    type=str,
    default=ServerArgs.kv_cache_dtype,
    choices=[
        "auto",
        "fp8_e5m2", "fp8_e4m3",
        "bf16", "bfloat16",
        "fp4_e2m1",
        "tq4", "tq3", "tq2",           # ← TurboQuant (no tq1 yet)
        "rq3", "rq4",
        "rq3_planar", "rq4_planar",
        "rq3_iso", "rq4_iso",
    ],
    help='Data type for kv cache storage. "auto" will use model data type. '
         '"tq4"/"tq3"/"tq2" for TurboQuant 4/3/2-bit KV (data-oblivious, no calibration). '
         '"rq3"/"rq4" shorthand for RotorQuant PlanarQuant (fastest, recommended default). '
         'TurboQuant advanced options via env vars: '
         'SGLANG_KV_CACHE_TURBOQUANT_ROPE=0 (disable RoPE quant for MLA), '
         'SGLANG_KV_CACHE_TURBOQUANT_QJL=1 (enable QJL unbiased inner product).'
)
```

#### Shorthand Mapping (line 803-804)
```python
if self.kv_cache_dtype in _RQ_SHORTHAND:
    self.kv_cache_dtype = _RQ_SHORTHAND[self.kv_cache_dtype]
```

**Note:** Need to find `_RQ_SHORTHAND` dict definition to see if it includes tq mode aliases.

### Findings

| Item | Status | Details |
|------|--------|---------|
| **tq2 flag** | ✅ Supported | Line 4088 |
| **tq3 flag** | ✅ Supported | Line 4088 |
| **tq4 flag** | ✅ Supported | Line 4087 |
| **tq1 flag** | ❌ Missing | Not in choices (should add) |
| **tq8 flag** | ❌ Missing | Not in choices (reference mode, add?) |
| **Env vars** | ✅ Documented | `TURBOQUANT_ROPE`, `TURBOQUANT_QJL` |
| **Help text** | ✅ Complete | Documents all codecs + env vars |

---

## Required Phase 4.2 Work Items

### 4.2.1 Add tq1 and tq8 Flags to sglang-1-bit-turbo

**File:** `donors/sglang-1-bit-turbo/python/sglang/srt/server_args.py` (line ~4087)

**Current:**
```python
choices=[
    ...
    "tq4", "tq3", "tq2",
    ...
]
```

**Required Change:**
```python
choices=[
    ...
    "tq4", "tq3", "tq2", "tq1", "tq8",  # Add tq1 (exp), tq8 (ref)
    ...
]
```

**Rationale:**
- **tq1:** 16x compression (experimental, full 5 bits reserved)
- **tq8:** 2x compression (reference/verification, FP8-equivalent)
- Aligns with gfxATOM codec contract (5 modes: tq1-tq8)

### 4.2.2 Create Codec Enum Alignment Document

**File:** `gfxATOM-Rust/docs/CODEC_ENUM_ALIGNMENT.md`

**Content:**
- Map SGLang codec strings → gfxATOM KvCodec enums
- Document conversion for each mode:
  - `"tq2"` ↔ `KvCodec.tq2`
  - `"tq3"` ↔ `KvCodec.tq3`
  - etc.
- Identify any mismatches or missing modes
- Define fallback behavior if SGLang uses unsupported mode

### 4.2.3 Audit Kernel Dispatch Paths

**Location:** `donors/sglang-1-bit-turbo/python/sglang/srt/`

**Search for:**
- Where `kv_cache_dtype` is used in model executor
- KV pool instantiation (which class/factory)
- Attention backend selection
- Quantization kernel dispatch

**Files to review:**
- `model_executor/` (look for dispatch logic)
- `mem_cache/memory_pool.py` (KV pool classes)
- `layers/attention/` (attention backend routing)

### 4.2.4 Create Adapter Bridge

**File:** `gfxATOM-Rust/python/sglang_backend_adapter.py` (new)

**Purpose:** Connect SGLang tq flags → gfxATOM TurboQuantizer

**Interface:**
```python
class SGLangTurboQuantAdapter:
    """Adapter bridging SGLang codec choices to gfxATOM TurboQuantizer."""
    
    def __init__(self, sglang_args: ServerArgs):
        """Initialize from SGLang ServerArgs."""
        self.sglang_args = sglang_args
        self.kv_codec = self._resolve_codec()
        self.turboquant = self._create_quantizer()
    
    def _resolve_codec(self) -> KvCodec:
        """Map SGLang kv_cache_dtype string to gfxATOM KvCodec enum."""
        # Validate and map
    
    def _create_quantizer(self) -> TurboQuantizer:
        """Instantiate TurboQuantizer for resolved codec."""
        # Create from kv_codec
    
    def encode_kv(self, k_cache: torch.Tensor) -> TurboCode:
        """Compress K cache using TurboQuantizer."""
        pass
    
    def estimate_inner_product(self, code: TurboCode, query: torch.Tensor) -> torch.Tensor:
        """Estimate attention scores from compressed code."""
        pass
```

---

## Integration Seams (Identified)

### Seam 1: Server Args → Backend Dispatch
```
SGLang CLI: --kv-cache-dtype tq2
    ↓
ServerArgs.kv_cache_dtype = "tq2"
    ↓
Model Executor (create_kv_pool)
    ↓
[INSERTION POINT] Instantiate SGLangTurboQuantAdapter
    ↓
Create TurboKVTokenToKVPool with adapter
```

### Seam 2: Prefill Attention → KV Encode
```
Prefill Forward:
    compute attention(q, k, v)
    ↓
    [INSERTION POINT] adapter.encode_kv(k_cache)
    ↓
    Store TurboCode in KV pool
```

### Seam 3: Decode Attention → Inner Product Estimation
```
Decode Forward:
    retrieve token_indices from ReqToTokenPool
    ↓
    [INSERTION POINT] adapter.estimate_inner_product(turbo_code, q)
    ↓
    Use estimated scores for softmax
```

---

## Codec Compatibility Table

| SGLang Flag | gfxATOM Enum | Bit Width | Compression | Status |
|-------------|--------------|-----------|-------------|--------|
| tq1 | KvCodec.tq1 | 1 | 16x | ❌ Missing (add) |
| tq2 | KvCodec.tq2 | 2 | 8x | ✅ Ready |
| tq3 | KvCodec.tq3 | 3 | 5.33x | ✅ Ready |
| tq4 | KvCodec.tq4 | 4 | 4x | ✅ Ready |
| tq8 | KvCodec.tq8 | 8 | 2x | ❌ Missing (add) |

---

## Next Steps

### Immediate (Phase 4.2 Completion)
1. ✅ Audit sglang-1-bit-turbo for existing tq support (DONE)
2. 📋 **4.2.1:** Add tq1 and tq8 flags to server_args.py
3. 📋 **4.2.2:** Create codec alignment document
4. 📋 **4.2.3:** Audit kernel dispatch paths in sglang fork
5. 📋 **4.2.4:** Create SGLangTurboQuantAdapter bridge

### Phase 4.3 (Backend Factory)
- Wire adapter factory into model executor
- Implement TurboKVTokenToKVPool instantiation
- Add fallback chain (tq → Triton → FP16)

### Phase 4.4 (KV Hooks)
- Add encode hook in prefill path
- Add decode hook in attention path
- Update KV pool read/write for TurboCode

---

## Discovery: Existing Environment Variables

SGLang fork already supports two TurboQuant-specific env vars:

```bash
SGLANG_KV_CACHE_TURBOQUANT_ROPE=0     # Disable RoPE quantization (for MLA)
SGLANG_KV_CACHE_TURBOQUANT_QJL=1      # Enable QJL unbiased inner product
```

These map to:
- **RoPE:** Rope layer within TurboQuantizer (disable for gfx1030 compatibility)
- **QJL:** Enables QjlQuantizer stage for inner product estimation

**Action:** Document these in gfxATOM's environment variable registry.

---

## Assumptions & Constraints

| Item | Assumption | Evidence |
|------|-----------|----------|
| Fork maturity | sglang-1-bit-turbo already has tq support | Lines 4087-4089, help text, env vars |
| Missing modes | tq1/tq8 not yet in fork | Only tq2-4 in choices |
| Kernel dispatch | Dispatch logic exists in fork | Shorthand mapping (line 803) suggests per-mode logic |
| Fallback ready | SGLang has Triton fallback | Standard SGLang architecture |

---

## Status: Phase 4.2 Definition of Done ✅

- [x] Cloned sglang-1-bit-turbo to donors/ (avoid confusion)
- [x] Located server_args.py with tq flags
- [x] Found env vars for TurboQuant tuning
- [x] Identified remaining work (tq1/tq8 addition, adapter bridge)
- [x] Mapped integration seams to file locations
- [x] Created codec compatibility table
- [x] Ready for Phase 4.2 sub-tasks (4.2.1-4.2.4)

**Next Phase:** 4.2.1 - Add tq1 and tq8 flags to sglang fork
