# Changelog

All notable changes to gfxATOM are documented in this file.

## [Unreleased] — 2026-05-19

### Verified — Rebuild audit (ATOM / gfxATOM-Rust / build-kernels)

- Audited `/home/local/ai/build/wip/ATOM` against active WIP dependencies and recent ROCm wheel/kernel forward-ports.
- Confirmed local dirt is limited to `rust_bindings/target/flycheck0/*` generated outputs (no source drift requiring rebuild).
- No ATOM source rebuild was required from this audit pass.

## [Unreleased] — 2026-05-13

### Added — Kernel Backend Expansion (Wave 1)

- **Auralis audio optimization merge resolution** (`atom/audio/chatterbox/engine.py`): Resolved PR #27 against current `main` by preserving preallocated CPU ONNX token and attention-mask buffers while keeping the Auralis audio post-processing and benchmark additions.

- **Rust File Walker** (`atom_rust.find_files`): Synced the SGLang `ignore::WalkBuilder` recursive file walker into gfxATOM's existing PyO3 extension, added a Python `os.walk` fallback helper, and wired recursive Python-file discovery through it. This targets faster traversal of large Hugging Face/model cache trees while preserving hidden-file visibility and symlink traversal for snapshot layouts.

- **Chatterbox ONNX CPU/Q8 Helpers** (`atom/audio/chatterbox/`): Centralized ONNX Runtime CPU session tuning with physical-core default intra-op threads, added Q8 sidecar artifact resolution/conversion helpers, exposed `--tts-onnx-variant` and `--tts-onnx-threads`, and added a `quantize_q8` CLI that writes optimized sidecars without mutating Hugging Face snapshots.

- **Audio Runtime API** (`atom/audio/runtime.py`): Promoted ONNX Runtime CPU session tuning into the public gfxATOM audio package with reusable `OnnxCpuRuntimeConfig` helpers and compatibility exports for Chatterbox callers.

- **Chatterbox Streaming Conditioning Cache** (`atom/audio/chatterbox/service.py`): Cached default reference conditioning so repeated short TTS chunks avoid re-running the speech encoder, reducing measured Turbo ONNX chunk latency by roughly 0.3-0.5s per chunk on the local RX 6700 XT host.

- **Chatterbox vLLM/cbx Assimilation** (`atom/audio/chatterbox/vllm_backend.py`): Added an optional `atom_vllm` Chatterbox backend mode that treats `/home/local/ai/engines/chatterbox-vllm` as a donor runtime, prepares its T3 tokenizer/model layout in an ATOM cache, exposes vLLM-style batch/chunk/CFG/diffusion/sampling knobs through `/v1/audio/speech`, and falls back to the existing ONNX/HF engine when the vLLM runtime is not importable. The default Chatterbox voice reference now points to DEMERZEL `af_bella.wav` for US female tuning trials, and the benchmark harness pins playback to the analog line-out sink instead of the headset path.

- **gfxATOM ROCm Build Bridge** (`pyproject.toml`, `.venv/lib/python3.12/site-packages/bridge.pth`): Added the Rust extension build helpers as `uv` dev dependencies and repathed the local gfxATOM venv through `/home/local/ai/.venv` so `uv run` resolves ROCm PyTorch, compatible AITER, Triton, and gfxGRAPH from the canonical AI bridge while preserving the local editable ATOM install.

- **Chatterbox-vLLM Runtime Enablement** (`atom/audio/chatterbox/vllm_backend.py`): Built and installed the local `/home/local/ai/forks/vllm` ROCm fork into the gfxATOM venv with `uv`, targeting `gfx1030` and ROCm only. Added ATOM-side compatibility shims for donor Chatterbox-vLLM imports that moved in newer vLLM internals (`SamplingMetadata`, multimodal processing aliases, tokenizer registry), and installed the minimal Chatterbox audio dependencies needed for `chatterbox_vllm.tts.ChatterboxTTS` to import without adding NVIDIA packages.

- **Chatterbox-vLLM Worker Compatibility** (`atom/audio/chatterbox/vllm_backend.py`, `atom/plugin/vllm/register.py`): Registered Chatterbox T3 inside spawned vLLM worker processes, adapted donor tokenizers/processors/model hooks to current vLLM APIs, defaulted T3 to FP16 on RDNA2, and disabled ROCm skinny/AITER Triton GEMM defaults for this path after live trials reached prompt execution but exposed gfx1030 launch failures in the unquantized GEMM decode path.

- **Chatterbox Kernel Stability Gate** (`atom/audio/chatterbox/kernel_candidates.py`, `atom/audio/chatterbox/vllm_backend.py`): Classified the local donor kernel trees under `/home/local/ai/build/kernels`, `/home/local/ai/build/wip`, and `/home/local/ai/engines` for Chatterbox T3 use. RDNA2/gfx1030 `atom_vllm` now defaults to conservative FP16 eager execution with skinny/AITER GEMM disabled unless `ATOM_CHATTERBOX_EXPERIMENTAL_GEMM=1` is set after a local Chatterbox-shape microbench. Runtime generation failures now report `requested_backend=atom_vllm` and fall back to the configured ONNX/HF engine when available.

- **Chatterbox Backend Argument Hardening** (`atom/entrypoints/openai/serving_speech.py`, `atom/audio/chatterbox/service.py`): Made `extra_params.backend` the final backend override, preserved safe filtering for engines without `**kwargs`, and propagated `exaggeration` through repeated ONNX token embedding steps instead of resetting to `0.5` after the initial prompt.

- **LFM2.5-Audio llama.cpp Bridge** (`atom/audio/lfm25_audio.py`, `scripts/lfm25_audio_e2e.py`): Added an ATOM bridge for the local `llama.cpp-audio-max` runtime with Q8/F16 GGUF artifact resolution, OpenAI SSE parsing for text/audio deltas, TTS registration under `lfm25_audio` / `lfm2.5-audio`, and `/v1/audio/transcriptions` plus `/v1/audio/transcribe` routes backed by LFM2.5-Audio `input_audio`. The proof script can run quantized and unquantized trials, save metrics/WAVs, and play results over the configured line-out sink without installing packages.

- **Spec Decode HIP Crash Fix**: Added try/except guard with CPU `index_select` fallback in `model_runner.py` for GPU-side fancy indexing that triggers `AcceleratorError` on RDNA2 (gfx1030) during speculative decoding's draft token gathering.

- **SnakeBeta Activation** (`atom/model_ops/rdna2/snake_activation.py`): Ported fused Triton SnakeBeta activation kernel from vLLM-Omni. Used by Qwen3-TTS / Code2Wav audio decoders. Features precomputed exp cache, auto-fallback to eager PyTorch, and inference-mode Triton auto-detection.

- **AITER Triton Flash Attention** (`atom/model_ops/attentions/aiter_triton_flash/`): Integrated the pre-tuned AMD Triton flash attention module with full FA2/FA3 forward and backward interfaces. Includes RDNA2 architecture detection, extended-context autotune configs, and configurable `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON` override.

- **LoRA FP8 Triton Ops** (`atom/model_ops/lora_triton/`): Ported 7 FP8-precision LoRA Triton kernels from vLLM (`lora_shrink_fp8`, `lora_expand_fp8`, `fused_moe_lora_fp8`, `fp8_kernel_utils`, `kernel_utils`, `lora_kernel_metadata`, `utils`). Created `_compat.py` shim to replace all vLLM dependencies (triton_utils, logger, platforms, distributed, torch_utils).

- **Layernorm Gated** (`atom/model_ops/rdna2/layernorm_gated.py`): Ported fused RMSNorm+Gating Triton kernel from vLLM's Mamba ops. Supports grouped normalization with optional gating (SiLU), bias, and both LayerNorm and RMSNorm modes.

- **gfxGRAPH Integration** (`atom/model_ops/gfxgraph.py`): Created ATOM integration layer for the gfxGRAPH Rust bridge (`gfxgraph_rs`). Provides lazy import, `ATOM_ENABLE_GFXGRAPH` environment variable gating, and `create_atom_bucket_router()` factory with standard ATOM batch size buckets. Fixed BucketSelector → BucketRouter alias in `gfxgraph_rs/src/lib.rs`.

- **Environment Variable**: Added `ATOM_ENABLE_GFXGRAPH` to `atom/utils/envs.py` (default: disabled).
