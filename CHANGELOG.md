
## Wave 33B Phase 4.3: TurboQuant/RotorQuant Routing Canonicalization

**Date:** 2026-05-17  
**Work:** Canonicalized TurboQuant/RotorQuant codec routing across Rust and Python adapter layers  
**Status:** In progress

### Changes

1. **Expanded codec family helpers in Rust**
   - Added TurboQuant/RotorQuant family predicates and bit-width helpers in `rs_kv_quant_contracts`
   - Routed `tq*`, `rq*_planar`, `rq*_iso`, FP8, and INT8 through `CodecAdapterRegistry`

2. **Aligned Python adapter registry and SGLang bridge**
   - Added RotorQuant iso modes plus fp8/int8 parity in `python/kv_codec_adapters.py`
   - Updated SGLang backend and AutoQuant summaries to emit rotor-aware backend chains

3. **Extended runtime capability helpers**
   - Added quantization-family capability helpers to `rs_atom_engine_profile` and its Python mirror
   - Added regression coverage for RotorQuant and backend-chain summaries

### Integration Gates Status

- [x] Rust codec family helpers added
- [x] Rust AutoQuant backend summary routes RotorQuant
- [x] Python codec registry supports RotorQuant iso modes
- [x] Python SGLang adapter emits rotor-aware fallback chains
- [ ] Full Rust/Python test validation
- [ ] Rust/Python profile parity verification after helper additions

### Next Steps

1. Run targeted Rust and Python tests for codec routing and runtime profile parity.
2. Update the Phase 4.3 plan checkpoint after validation.

## Wave 33 Phase 2 Follow-Up: Kernel Improvements Identification

**Date:** 2026-05-17  
**Work:** Identified high-priority kernel improvements from upstream ATOM integration wave  
**Status:** Phase 2 action planning (integration staging)  

### Changes

1. **Created kernel improvements tagging document**
   - File: `docs/kernel-improvements/wave33-phase2-kernel-tagging.md` (moved to manifests)
   - Identified high-priority kernels: FlashInfer CUTLASS, RDNA2 HIP, TurboQuant/RotorQuant
   - Linked to upstream commits: 3b60317, a4da908, 64f7808, d526c3a

2. **Model support additions (no code changes required)**
   - DeepSeek-v4: Text generation with parallel head optimization
   - Kimi K2.5: Eagle3 speculative decoding
   - LFM2/LFM2.5: Audio models (STT/TTS)
   - Fish Speech S2 Pro: TTS audio model
   - VoxCPM2: TTS audio model
   - Note: All models are runtime-driven; gfxATOM-Rust does not enumerate models

3. **Audio layer coordination deferred to next sprint**
   - Documented DEMERZEL boundary (owns orchestration/routing)
   - Upstream ATOM audio improvements are complementary
   - Pattern: improvements flow through ATOM runtime → gfxATOM backend → DEMERZEL routing

### Integration Gates Status

- [x] Upstream ATOM security fix (SafeUnpickler) applied
- [x] gfxATOM-Rust FP8 KV alignment contract added
- [x] Warmup initialization support added to Rust profile
- [x] Full Rust test suite passing (35/35 tests)
- [ ] RDNA2 kernel validation for gfx1030
- [ ] TurboQuant integration test harness
- [ ] Upstream ATOM KV warmup init applied to runtime

### Next Steps

1. **Wave 33A** — Kernel improvement sequencing
   - Verify RDNA2 HIP kernels in canonical collection
   - Tag high-priority kernels (FlashInfer, RDNA2, TurboQuant)
   - Update canonical index with integration targets

2. **Wave 33B** — Audio coordination planning
   - Create task: "Audio layer integration with DEMERZEL"
   - Define coordination boundary and information flow

3. **Phase 3** — Runtime integration
   - KV warmup hook implementation
   - Kernel dispatch routing for RDNA2
   - TurboQuant codec registry integration
