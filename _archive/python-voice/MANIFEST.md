# Archived Python Voice (superseded by the Rust voice path)

Archived 2026-06-19 during the voice→Rust consolidation. The live voice is pure Rust
(`rs_chatterbox_engine` TTS, `rs_asr_serving` whisper/moonshine, `rs_vad_core` silero,
`rs_wake_word` openwakeword) served by `engines/atom-rs/bin/{voice_serving_bin,speech_serving_bin}`.
These Python sources/forks are superseded.

## Kept here (the unique part of `build/wip/chatterbox`)
- `chatterbox/` — our `build/wip/chatterbox` WIP **source** (`src/` + `benchmarks/` + configs + its
  `README.md`/`CHANGELOG.md`). All 4 markdown files that were under the deleted tree are preserved
  byte-for-byte here — see [`MARKDOWN-RECOVERY-AUDIT.md`](MARKDOWN-RECOVERY-AUDIT.md) for the hash proof.

## Deleted from `build/wip/chatterbox`, NOT lost (verified 2026-06-19)
- **`models/` (15G)** — this was the HF `snapshot_download` / benchmark-scratch **input**, NOT the
  customized models. The customized Rust models live in **`models/chatterbox-rs`** (INTACT, 20G) and are
  **BLAKE3-verified bit-perfect** against their `blake3.json` (turbo 47/47, base 23/23); both the **turbo**
  and **base/tts** customized models were proven to load and synthesize real audio. A second copy + the
  original `.pt` checkpoints are in `.quarantine/2026-06-12/wip/chatterbox-rust`.
- **`donors/`** — donor clones; the harvested value is retained in `build/donors/harvested/`, and the
  clones are upstream-recoverable (`chatterbox-vllm`, `CosyVoice`, `GPT-SoVITS`).

> ⚠️ **Lesson (user directive 2026-06-19):** never delete models or engines without explicit gated
> approval; "recoverable from HuggingFace" is NOT a justification to delete a *customized* model, and a
> running service is not proof a model is intact. The earlier wording here understated that — corrected.
