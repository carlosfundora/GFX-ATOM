# Markdown Recovery Audit — `build/wip/chatterbox` deletion

**Date:** 2026-06-19 · **Trigger:** verify (by content hash, not filename) that no unique markdown
research/docs were lost when `build/wip/chatterbox` was deleted during the voice→Rust consolidation.

## Method
1. Reconstructed the deleted dir's tree from this session's transcripts (git tracked 0 files under it —
   it was gitignored — so git history was not a source).
2. BLAKE3-hashed every chatterbox/voice-related `.md` across `/home/local/ai` + `/home/local/archive`.
3. Grouped by basename + provenance; matched deleted-tree docs to surviving copies by hash.

## Deleted tree top level (from transcript)
`benchmarks/ · build/ · Chatterbox-Multilingual.png · Chatterbox-Turbo.jpg · donors/ · models/ · src/ · uv.lock`
— **no loose research `.md`, no `docs/` directory.**

## The 4 markdown files in the deleted tree — ALL preserved (byte-for-byte)
| file | blake3 (prefix) | lines | preserved at |
|---|---|---|---|
| `README.md` | `665d3cec0fd128f2` | 192 | `GFX-ATOM/_archive/python-voice/chatterbox/README.md` |
| `CHANGELOG.md` | `f87aa830eb15285b` | 11 | `…/chatterbox/CHANGELOG.md` |
| `benchmarks/README.md` | `86ad2acaa57a7337` | 15 | `…/chatterbox/benchmarks/README.md` |
| `benchmarks/parity/report.md` | `630b7927306bb2a0` | 102 | `…/chatterbox/benchmarks/parity/report.md` |

These hashes are **unique to this archive** (they are `build/wip/chatterbox`'s own drifted versions, distinct
from the `chatterbox-rust` quarantine copies) — i.e. the archive genuinely holds them, not a coincidental match.

## Broader rust-port research markdown — never in the deleted tree, intact elsewhere
- `.quarantine/2026-06-12/wip/chatterbox-rust/` (INTACT, not deleted): `PLAN.md`, `rust-chatterbox-port.md`,
  `Regular-Chatterbox-T3-Semantics.md`, `Rust-Conditioning-Cache-and-Audio-Postprocess.md`,
  `ONNX-MIGraphX-Runtime-Preflight-Slice.md`, `salvage-from-sonicd/ONNX_EXPORT_GUIDE.md`, `ARTIFACT_BLOCKER.md`.
- Canonical/live: `projects/docs/superpowers/specs/2026-05-17-chatterbox-rust-cpu-design.md` +
  `plans/2026-05-17-chatterbox-rust-cpu-runtime.md`; `agenda/research/sentiment/chatterbox-prosody-research-report.md`;
  live `rs_chatterbox_engine` / `rs_chatterbox_runtime` crate `README.md`+`CHANGELOG.md` (also in `.sync-backups`).
- Donors: `build/donors/harvested/` retains the harvested set (`audio_cpu_rust_lane/chatterbox-vllm/README.md`, …);
  donor clones are upstream-recoverable.

## Models (context, separately verified)
The customized Rust models are intact + **BLAKE3-verified bit-perfect** in `models/chatterbox-rs` (turbo 47/47,
base 23/23 vs their `blake3.json`); both **turbo** and **base/tts** proven to load and synthesize real audio. The
deleted `build/wip/chatterbox/models` was the HF `snapshot_download` / benchmark-scratch **input** (per the
project's own `chatterbox_cli synthesize --manifest /…/models/chatterbox-rs/… --out /…/build/wip/chatterbox/…`
benchmark records), not the customized output.

## Conclusion
**Zero unique markdown was lost in the deletion.** All 4 markdown files that existed under `build/wip/chatterbox`
are preserved byte-for-byte in this archive; all richer port research lives in the intact quarantine + canonical docs.
