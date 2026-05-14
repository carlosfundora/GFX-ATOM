import io
from pathlib import Path

import numpy as np
import soundfile as sf

from atom.audio.lfm25_audio import (
    LFM25AudioEngine,
    build_lfm25_server_command,
    resolve_lfm25_audio_paths,
)


def _touch_lfm_files(root: Path, *, f16: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "LFM2.5-Audio-1.5B-Q8_0.gguf").write_text("q8")
    if f16:
        (root / "LFM2.5-Audio-1.5B-F16.gguf").write_text("f16")
    (root / "mmproj-LFM2.5-Audio-1.5B-F16.gguf").write_text("mm")
    (root / "vocoder-LFM2.5-Audio-1.5B-Q8_0.gguf").write_text("voc")
    (root / "tokenizer-LFM2.5-Audio-1.5B-Q8_0.gguf").write_text("tok")


def test_resolve_lfm25_audio_paths_prefers_requested_precision(tmp_path):
    _touch_lfm_files(tmp_path)

    q8 = resolve_lfm25_audio_paths(tmp_path, precision="q8")
    f16 = resolve_lfm25_audio_paths(tmp_path, precision="f16")

    assert q8.model.name.endswith("Q8_0.gguf")
    assert f16.model.name.endswith("F16.gguf")
    assert q8.mmproj.name.endswith("F16.gguf")


def test_lfm25_server_command_uses_audio_artifacts(tmp_path):
    _touch_lfm_files(tmp_path)
    paths = resolve_lfm25_audio_paths(tmp_path)

    cmd = build_lfm25_server_command(paths, port=30123, n_gpu_layers=0)

    assert "--model" in cmd
    assert str(paths.model) in cmd
    assert "--mmproj" in cmd
    assert str(paths.mmproj) in cmd
    assert "--model-vocoder" in cmd
    assert "--tts-speaker-file" in cmd
    assert "30123" in cmd


class FakeLFMClient:
    def text_to_speech(self, text, *, max_tokens=512):
        buf = io.BytesIO()
        wav = np.zeros(2400, dtype=np.float32)
        sf.write(buf, wav, 24000, format="WAV", subtype="PCM_16")
        return {"wav_bytes": buf.getvalue(), "audio_chunks": 1}


def test_lfm25_audio_engine_returns_atom_speech_tuple():
    engine = LFM25AudioEngine(FakeLFMClient())

    wav, metrics = engine.generate("hello", max_tokens=8)

    assert wav.dtype == np.float32
    assert engine.sample_rate == 24000
    assert metrics["backend"] == "lfm2.5-audio"
    assert metrics["audio_chunks"] == 1
