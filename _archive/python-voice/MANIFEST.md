# Archived Python Voice (superseded by the Rust voice path)

Archived 2026-06-19 during the voice→Rust consolidation. The live voice is pure Rust
(`rs_chatterbox_engine` TTS, `rs_asr_serving` whisper/moonshine, `rs_vad_core` silero,
`rs_wake_word` openwakeword) served by `engines/atom-rs/bin/{voice_serving_bin,speech_serving_bin}`.
These Python sources/forks are superseded.

## Kept here (non-recoverable — no git remote)
- `chatterbox/` — our `build/wip/chatterbox` WIP **source** (`src/` + `benchmarks/` + configs). The
  26G original was 15G regenerable models + 4.8G donor clones (deleted); only this source was unique.

## Deleted, recoverable from upstream (NOT archived — already on GitHub)
- `engines/chatterbox-vllm` → re-clone `https://github.com/randombk/chatterbox-vllm.git`
- `audio/CosyVoice` → re-clone `https://github.com/FunAudioLLM/CosyVoice.git`  (abandoned TTS alt)
- `audio/GPT-SoVITS` → re-clone `https://github.com/RVC-Boss/GPT-SoVITS.git`  (abandoned TTS alt, 4.3G)
- `build/wip/chatterbox/models` (15G) → re-downloadable from HuggingFace; the live Rust uses `models/chatterbox-rs`.
