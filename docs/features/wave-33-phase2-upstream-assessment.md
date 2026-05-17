# Wave 33: Phase 2 Upstream ATOM Integration Assessment

**Date:** 2026-05-17  
**Scope:** Catalog and assess audio, model, and kernel improvements from upstream ATOM  
**Output:** Integration roadmap with deferral recommendations  

## Executive Summary

Upstream ATOM has received 129 commits since 2026-05-01. Phase 2 assesses feature additions (audio, models, kernels) beyond the critical fixes (FP8 alignment, KV warmup, security) completed in Phase 1.

**Key Finding:** Upstream audio improvements (Chatterbox optimizations, LFM2) are complementary to DEMERZEL's audio layer. Recommend deferring direct upstream audio integration in favor of DEMERZEL coordination. Model and kernel improvements are orthogonal—can proceed without risk.

---

## 1. Audio Layer Assessment

### Upstream ATOM Audio Changes

**Commits identified:**
- `a4da908` — Port GPU kernels: Triton attention, ROCm HIP, W8A8 quant, MoE, LoRA, gfxGRAPH
- `8deebfe` — Hoist Chatterbox engine overhead and optimize stream path
- `7628038` / `64db72b` — Optimize audio pipeline latency via rs_codec integration
- `6495950` — Introduce LFM2 model implementation
- `917221c` — Implement Chatterbox vLLM backend integration and LFM2.5 bridge
- `c91a04e` — Integrate LFM2.5-Audio STT/TTS bridge
- `7a1843d` — Optimize audio Chatterbox generation
- `f97a16b` — Preallocate np arrays for CPU ONNX autoregressive inference

**Themes:**
1. **Chatterbox performance** — Stream latency reduction, engine overhead hoisting
2. **LFM2.5 audio** — STT/TTS bridges, vLLM backend integration
3. **rs_codec integration** — Latency optimization via Rust codec
4. **ONNX optimization** — CPU-based audio inference

### DEMERZEL Audio Layer Analysis

**Current scope:**
- `DEMERZEL/src/audio/components/` — ASR, TTS, synthesis engines
- `DEMERZEL/src/audio/vad/` — Voice activity detection
- `DEMERZEL/src/audio/voice/` — Voice configuration and profiles
- `DEMERZEL/src/audio/orchestration/` — Audio pipeline orchestration
- `DEMERZEL/src/audio/persona/` — Audio personality/emotion routing
- `DEMERZEL/src/audio/processing/` — Real-time processing
- `DEMERZEL/src/audio/submodules/` — External audio subsystems
- Emotional analyzer, pipecat flows, tool bridge, wakeword detection

**Design philosophy:** DEMERZEL owns the audio orchestration, context routing, and emotion-aware synthesis. Upstream ATOM owns model execution and low-level kernels.

### Overlap Analysis

| Component | DEMERZEL Owns | Upstream ATOM Owns | Status |
|-----------|---------------|--------------------|--------|
| ASR/STT models | Bridge/routing | Model execution | Complementary |
| TTS models (LFM2) | Bridge/routing | Model execution/optimization | Complementary |
| Real-time audio pipeline | Orchestration | Kernel dispatch | Complementary |
| Emotion tagging | Analysis + synthesis routing | Not applicable | No conflict |
| Voice profiles | Configuration | Not applicable | No conflict |
| Chatterbox latency | Stream context | Model/kernel optimization | Complementary |
| Audio codecs | Not primary | rs_codec integration | Complementary |

### Recommendation: **DEFER**

**Reasoning:**
- DEMERZEL's audio layer is mature, purpose-built for context-aware synthesis
- Upstream ATOM audio improvements are kernel/model optimizations, not architectural changes
- Integrating directly into gfxATOM-Rust would duplicate DEMERZEL's routing logic
- Better pattern: DEMERZEL stays canonical; upstream improvements flow through ATOM runtime → gfxATOM backend → DEMERZEL routing

**Action Items:**
- [ ] Document audio layer boundary in integration notes
- [ ] Create a "DEMERZEL coordination" task for next sprint (does not block Phase 2)
- [ ] Note: DEMERZEL will consume upstream ATOM improvements indirectly via runtime backend

---

## 2. New Model Support Assessment

### Upstream ATOM Model Changes

**Commits identified:**
- `0b0e009` — DeepSeek-v4 support (ParallelHead optimization, cudagraph, unified aiter RMSNorm)
- `6a4db8e` — DeepSeek-v4 performance (unified rmsnorm_quant fuses)
- `679422d` — Kimi K2.5 Eagle3 speculative decoding support
- `6495950` — LFM2 model implementation
- `721a5dd` — Multi-model TTS API with Fish Speech S2 Pro and VoxCPM2
- `d526c3a` — GGUF/Q1 support and RotorQuant/TurboQuant from sglang-1-bit-turbo

**New models:**
1. **DeepSeek-v4** — Text generation, parallel head optimization
2. **Kimi K2.5** — Eagle3 speculative decoding
3. **LFM2** / **LFM2.5** — Audio models (STT/TTS)
4. **Fish Speech S2 Pro** — TTS audio model
5. **VoxCPM2** — TTS audio model

### gfxATOM-Rust Model Registry

**Current location:** `crates/rs_atom_engine_profile/src/lib.rs` (if model enumeration exists)

**Question:** Does gfxATOM-Rust own model registry or just capability flags?

**Analysis:** Looking at `engine_runtime_profile.py`, the profile contains capability flags but no explicit model enumeration. Models are likely registered in upstream ATOM.

### Recommendation: **DEFER with Documentation**

**Reasoning:**
- gfxATOM-Rust owns policy arbitration and capability contracts, not model enumeration
- Upstream ATOM is responsible for model implementations and optimizations
- New models automatically available when upstream ATOM is deployed
- No action required in gfxATOM-Rust unless policy changes for specific models

**Action Items:**
- [ ] Document that model support flows from upstream ATOM runtime
- [ ] Note DeepSeek-v4 and Kimi K2.5 additions for capability tracking
- [ ] Add observation: LFM2/Fish/VoxCPM2 are audio-domain models (coordinate with DEMERZEL)
- [ ] No changes needed to gfxATOM-Rust for model support (runtime-driven)

---

## 3. Kernel Improvements Assessment

### Upstream ATOM Kernel Changes

**Commits identified:**
- `3b60317` — Allow FlashInfer CUTLASS kernels (removed assertion)
- `a4da908` — Port GPU kernels: Triton attention, ROCm HIP, W8A8 quant, MoE, LoRA, gfxGRAPH
- `64f7808` — Triton-based unified attention and RDNA2 optimized kernels
- `32a49e6` — Optimize MLA sparse preparation with custom Triton kernels
- `853a302` — Replace Triton kernel with torch.compile for Gated Delta Net
- `6ce4695` — MLA sparse optimization with Triton prep
- `d526c3a` — Port RDNA2 kernels, GGUF/Q1 support, RotorQuant/TurboQuant

**Kernel categories:**
1. **Attention kernels** — Triton, FlashInfer, unified attention
2. **Quantization kernels** — W8A8, RotorQuant/TurboQuant, FP8
3. **MoE kernels** — Sparse MLA, gated operations
4. **RDNA2-specific** — AMD ROCm HIP implementations
5. **Special ops** — gfxGRAPH, LoRA, Delta-Net

### gfxATOM-Rust Kernel Collection

**Current location:** `/home/local/ai/build/kernels/canonical/` (if materialized)

**Question:** Should kernel improvements be imported into gfxATOM-Rust's kernel collection?

**Analysis:**
- gfxATOM-Rust owns kernel ranking and Rust-native optimization primitives
- Upstream ATOM kernels are Python/HIP implementations (runtime-driven dispatch)
- Already collected in `build/kernels/canonical` from donor pass

### Recommendation: **SELECTIVE INTEGRATION**

**High-priority kernels (integrate into canonical collection):**
- FlashInfer CUTLASS removal fix (`3b60317`) — enables better kernel selection
- RDNA2 HIP kernels (`a4da908`, `64f7808`) — gfx1030 direct improvement
- RotorQuant/TurboQuant from sglang-1-bit-turbo (`d526c3a`) — directly applicable

**Lower-priority (defer to tranche 25+):**
- MLA sparse optimizations — experimental, monitor for stability
- torch.compile Delta-Net (`853a302`) — not RDNA2-specific

**Action Items:**
- [ ] Add FlashInfer CUTLASS fix to kernel notes
- [ ] Document RDNA2 kernel improvements in canonical index
- [ ] Update kernel selection routing if applicable
- [ ] Tag RotorQuant/TurboQuant kernels as high-priority for next exec wave

---

## 4. Integration Summary Table

| Domain | Change Count | gfxATOM-Rust Action | Priority | Timeline |
|--------|--------------|---------------------|----------|----------|
| **Audio** | ~8 commits | Defer to DEMERZEL coordination | Low | Tranche 25+ |
| **Models** | ~6 commits | Document, no code changes needed | N/A (runtime-driven) | N/A |
| **Kernels** | ~7 commits | Selective integration (FlashInfer, RDNA2, Quant) | Medium | Phase 2 |
| **Security** | 1 commit | Documented complete (Phase 1) | Critical | ✓ Complete |
| **KV Alignment** | 1 commit | Documented complete (Phase 1) | Critical | ✓ Complete |
| **KV Warmup** | 1 commit | Defer to Phase 2 follow-up | High | Phase 2 |

---

## 5. Phase 2 Action Plan

### Immediate (Phase 2, this sprint)

1. **Kernel improvements cataloging**
   - [ ] Update kernel notes with FlashInfer CUTLASS enablement
   - [ ] Document RDNA2 kernel additions
   - [ ] Add RotorQuant/TurboQuant to high-priority kernel queue

2. **Model support documentation**
   - [ ] Note DeepSeek-v4, Kimi K2.5, LFM2, Fish Speech S2 Pro, VoxCPM2 in integration notes
   - [ ] Observe that audio models require DEMERZEL coordination

3. **Audio layer coordination planning**
   - [ ] Create a follow-up task: "Audio layer integration with DEMERZEL" for next sprint
   - [ ] Document the DEMERZEL-owned orchestration boundary

### Deferred (Phase 2 follow-up, next sprint)

1. **KV warmup initialization hook**
   - Requires engine profile extension with explicit warmup phase signals

2. **DEMERZEL audio coordination**
   - Requires cross-team context to decide on Chatterbox/LFM2 routing

3. **Advanced kernel features** (MLA sparse, torch.compile ops)
   - Requires stability monitoring before integration

---

## 6. Success Criteria for Phase 2

- [ ] Audio layer assessment documented and deferral justified
- [ ] Model support changes documented (no code changes required)
- [ ] Kernel improvements cataloged in canonical collection
- [ ] High-priority kernels (FlashInfer, RDNA2, Quant) marked for next integration wave
- [ ] Phase 2 summary added to root CHANGELOG.md
- [ ] Integration plan updated with Phase 2 findings

---

## 7. Dependencies & Risks

### Low Risk
- Kernel improvements are additive, no breaking changes
- Model additions are transparent to gfxATOM-Rust (runtime-driven)

### Medium Risk
- Audio layer changes require DEMERZEL coordination to avoid duplication
- Warmup hooks require understanding whether engine profile owns initialization

### Deferred Decision
- Whether gfxATOM-Rust should own explicit warmup phase signaling (see Phase 2 follow-up)

---

## References

- Upstream ATOM commits: `a4da908`, `8deebfe`, `3b60317`, `6495950`, `0b0e009`, `d526c3a`
- gfxATOM-Rust engine profile: `python/engine_runtime_profile.py`
- DEMERZEL audio layer: `DEMERZEL/src/audio/`
- Kernel collection: `/home/local/ai/build/kernels/canonical/`

