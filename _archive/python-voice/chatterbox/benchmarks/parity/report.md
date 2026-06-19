# Chatterbox Rust vs Python parity + benchmark report

Generated: 2026-06-11T16:01:30-0500

CPU: AMD Ryzen 9 3900X 12-Core Processor  |  provider: CPUExecutionProvider (both sides)
torch: 2.6.0+cu124 (cpu (torch build is +cu124; no CUDA device on this AMD host))  |  onnxruntime: 1.26.0
wip/chatterbox commit: 65475944041e0fa8c6d528bbabaeba6adf8624c0 (toplevel /home/local/ai/build)
rust repo commit: 3ec5e28b191a19bd5b79b4dbc8a4d92fb5dc08e3
rust cli: /home/local/ai/projects/rust/target/release/chatterbox_cli

## Notes / caveats

- Rust side: greedy decode (deterministic). Python reference: stock sampling defaults (stochastic).
- Python 'base' reference is the multilingual v2 checkpoint (t3_mtl23ls_v2) run with language_id=en, because the English-base files (ve/t3_cfg/s3gen .safetensors) are absent from the local HF cache and downloads were disallowed. Cross-checkpoint parity is therefore best-effort for 'base'.
- logmel_cos: cosine over concatenated per-mel-bin (mean,std) of an 80-mel log spectrogram; global pooling, no DTW.
- speaker cosines use the local ONNX speech_encoder (fp16) speaker_embeddings[1,192] via onnxruntime CPU.
- WER via faster-whisper 'base' (CPU int8), text normalized to lowercase alphanumerics.
- Two upstream dtype bugs (NumPy-2 float64 promotion: norm_loudness gain in tts_turbo, and float64 wavs reaching S3Tokenizer.log_mel_spectrogram) monkeypatched in-process for the Python reference; repo source untouched.

## Variant: turbo

Prompts with rust output: 10 / with python ref: 10

### Gates (medians)

| gate | value | threshold | pass |
|---|---|---|---|
| duration_ratio | 0.9079 | 0.8–1.25 | True |
| spk_cos rust-vs-ref | 0.9131 | >= 0.8 | True |
| WER rust vs ref | rust 0.0 / ref 0.0 | rust <= ref+0.05 | True |

### Other medians

- logmel_cos: 0.9946
- spk_cos rust-vs-reference-voice (self-contained): 0.8915
- spk_cos ref-vs-reference-voice (context): 0.9063

### Bench (rust unless noted)

- median first_token_ms: 294.5
- median generate_ms: 5788.5
- median tokens/s (audio_samples/960 per generate_s): 13.3234
- median realtime factor (audio_s/generate_s): 0.5329
- median audio_s rust / ref: 2.94 / 3.2
- python ref median elapsed_s per prompt (CPU): 7.319

### Per-prompt

| # | dur ratio | logmel cos | spk r-vs-f | spk r-vs-voice | WER rust | WER ref | rust gen ms |
|---|---|---|---|---|---|---|---|
| 0 | 0.92 | 0.9821 | 0.6865 | 0.6321 | 0.0 | 0.0 | 1916 |
| 1 | 0.8889 | 0.9939 | 0.8545 | 0.8839 | 0.0 | 0.0 | 5779 |
| 2 | 0.9667 | 0.9979 | 0.9321 | 0.9053 | 0.0909 | 0.1818 | 7838 |
| 3 | 1.127 | 0.9905 | 0.9111 | 0.8777 | 0.0 | 0.0 | 5141 |
| 4 | 0.8958 | 0.9937 | 0.9622 | 0.9341 | 0.0 | 0.0 | 9731 |
| 5 | 1.2069 | 0.9964 | 0.8966 | 0.7106 | 0.0 | 0.0 | 4831 |
| 6 | 0.8522 | 0.997 | 0.9206 | 0.9155 | 0.0 | 0.0 | 5798 |
| 7 | 1.3448 | 0.9945 | 0.7933 | 0.8187 | 0.0 | 0.0 | 3585 |
| 8 | 0.8263 | 0.9947 | 0.9597 | 0.9308 | 0.0 | 0.0 | 10306 |
| 9 | 0.8636 | 0.9963 | 0.915 | 0.899 | 0.0 | 0.2857 | 6078 |

## Variant: base

Prompts with rust output: 10 / with python ref: 10

### Gates (medians)

| gate | value | threshold | pass |
|---|---|---|---|
| duration_ratio | 1.0265 | 0.8–1.25 | True |
| spk_cos rust-vs-ref | 0.9128 | >= 0.8 | True |
| WER rust vs ref | rust 0.0 / ref 0.0 | rust <= ref+0.05 | True |

### Other medians

- logmel_cos: 0.9979
- spk_cos rust-vs-reference-voice (self-contained): 0.8979
- spk_cos ref-vs-reference-voice (context): 0.9291

### Bench (rust unless noted)

- median first_token_ms: 336.5
- median generate_ms: 14192.0
- median tokens/s (audio_samples/960 per generate_s): 6.4028
- median realtime factor (audio_s/generate_s): 0.2561
- median audio_s rust / ref: 3.74 / 3.38
- python ref median elapsed_s per prompt (CPU): 15.395

### Per-prompt

| # | dur ratio | logmel cos | spk r-vs-f | spk r-vs-voice | WER rust | WER ref | rust gen ms |
|---|---|---|---|---|---|---|---|
| 0 | 1.0 | 0.9873 | 0.7408 | 0.6326 | 0.0 | 0.0 | 10174 |
| 1 | 1.2121 | 0.9985 | 0.8897 | 0.8984 | 0.0 | 0.0 | 12769 |
| 2 | 1.0142 | 0.9982 | 0.947 | 0.931 | 0.0909 | 0.0909 | 21864 |
| 3 | 0.8615 | 0.9973 | 0.9315 | 0.8974 | 0.1 | 0.0 | 10269 |
| 4 | 1.1111 | 0.9979 | 0.9655 | 0.9439 | 0.0 | 0.0 | 22455 |
| 5 | 1.069 | 0.9978 | 0.9042 | 0.7223 | 0.0 | 0.0 | 10384 |
| 6 | 1.2155 | 0.9983 | 0.8951 | 0.8945 | 0.0 | 0.0 | 19661 |
| 7 | 0.5714 | 0.9967 | 0.6417 | 0.7076 | 0.6667 | 0.0 | 6606 |
| 8 | 0.9688 | 0.994 | 0.9495 | 0.9353 | 0.0 | 0.0 | 20445 |
| 9 | 1.0388 | 0.9982 | 0.9214 | 0.9127 | 0.0 | 0.0 | 15615 |
