# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import numpy as np
import pytest

from atom.audio.chatterbox.vllm_backend import (
    DEFAULT_US_FEMALE_VOICE,
    ChatterboxAtomVllmEngine,
    _apply_rdna2_stability_env,
    _split_text,
    is_atom_vllm_available,
)
from atom.audio.protocol import AudioSpeechRequest, BatchSpeechRequest, SpeechBatchItem
from atom.entrypoints.openai.serving_speech import SpeechServing


class DummyFallback:
    sample_rate = 24000

    def __init__(self):
        self.loaded = False
        self.calls = []

    def load(self):
        self.loaded = True

    def generate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return np.zeros(240, dtype=np.float32), {
            "audio_duration": 0.01,
            "num_tokens": 1,
            "tok_per_sec": 1.0,
            "rtf": 1.0,
        }


class CaptureEngine:
    sample_rate = 24000

    def __init__(self):
        self.kwargs = None

    def generate(self, text, **kwargs):
        self.kwargs = kwargs
        return np.zeros(240, dtype=np.float32), {
            "audio_duration": 0.01,
            "num_tokens": 1,
            "tok_per_sec": 1.0,
            "rtf": 1.0,
        }


class CaptureBatchEngine(CaptureEngine):
    def __init__(self):
        super().__init__()
        self.batch_texts = None

    def generate_batch(self, texts, **kwargs):
        self.batch_texts = texts
        self.kwargs = kwargs
        return [
            (
                np.zeros(240, dtype=np.float32),
                {
                    "audio_duration": 0.01,
                    "num_tokens": 1,
                    "tok_per_sec": 1.0,
                    "rtf": 1.0,
                },
            )
            for _ in texts
        ]


class FailingLoadedModel:
    sr = 24000

    def generate(self, *args, **kwargs):
        raise RuntimeError("gfx1030 launch failure")


def test_default_us_female_voice_reference_is_configured():
    assert DEFAULT_US_FEMALE_VOICE.name == "af_bella.wav"
    assert DEFAULT_US_FEMALE_VOICE.exists()


def test_atom_vllm_availability_reports_missing_runtime_without_installing():
    ok, reason = is_atom_vllm_available("/path/that/is/not/used/when/vllm/is/missing")
    if not ok:
        assert reason
    else:
        assert reason is None


def test_atom_vllm_falls_back_when_runtime_is_unavailable(tmp_path):
    fallback = DummyFallback()
    engine = ChatterboxAtomVllmEngine(
        model_dir=str(tmp_path),
        source_dir=str(tmp_path / "missing-source"),
        fallback_engine=fallback,
    )

    engine.load()
    wav, metrics = engine.generate(
        "hello from atom",
        backend="atom_vllm",
        batch_size=4,
        chunk_chars=160,
        cfg_weight=0.5,
    )

    assert fallback.loaded
    assert wav.dtype == np.float32
    assert metrics["backend"] == "fallback"
    assert metrics["fallback_reason"]
    assert fallback.calls[0][0] == "hello from atom"


def test_atom_vllm_runtime_failure_falls_back_after_load(tmp_path):
    fallback = DummyFallback()
    engine = ChatterboxAtomVllmEngine(
        model_dir=str(tmp_path),
        source_dir=str(tmp_path),
        fallback_engine=fallback,
    )
    engine._model = FailingLoadedModel()

    wav, metrics = engine.generate(
        "fallback after runtime crash",
        cfg_weight=0.3,
        diffusion_steps=3,
    )

    assert wav.dtype == np.float32
    assert metrics["backend"] == "fallback"
    assert metrics["requested_backend"] == "atom_vllm"
    assert "gfx1030 launch failure" in metrics["fallback_reason"]
    assert fallback.calls[0][1]["exaggeration"] == 0.5


def test_rdna2_stability_gate_disables_unsafe_vllm_gemm(monkeypatch):
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    monkeypatch.delenv("ATOM_CHATTERBOX_EXPERIMENTAL_GEMM", raising=False)
    monkeypatch.delenv("VLLM_ROCM_USE_AITER_TRITON_GEMM", raising=False)
    monkeypatch.delenv("VLLM_ROCM_USE_SKINNY_GEMM", raising=False)

    _apply_rdna2_stability_env()

    assert os.environ["VLLM_ROCM_USE_AITER_TRITON_GEMM"] == "False"
    assert os.environ["VLLM_ROCM_USE_SKINNY_GEMM"] == "False"


def test_rdna2_stability_gate_respects_experimental_gemm_opt_in(monkeypatch):
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    monkeypatch.setenv("ATOM_CHATTERBOX_EXPERIMENTAL_GEMM", "1")
    monkeypatch.delenv("VLLM_ROCM_USE_AITER_TRITON_GEMM", raising=False)

    _apply_rdna2_stability_env()

    assert "VLLM_ROCM_USE_AITER_TRITON_GEMM" not in os.environ


def test_speech_request_exposes_chatterbox_vllm_knobs():
    req = AudioSpeechRequest(
        input="hello",
        backend="atom_vllm",
        batch_size=8,
        chunk_chars=240,
        cfg_weight=0.3,
        diffusion_steps=5,
        gpu_memory_utilization=0.72,
        enforce_eager=True,
        temperature=0.7,
        top_p=0.9,
        min_p=0.02,
    )

    assert req.backend == "atom_vllm"
    assert req.batch_size == 8
    assert req.diffusion_steps == 5


def test_serving_routes_backend_and_generation_knobs():
    serving = SpeechServing()
    default_engine = CaptureEngine()
    atom_engine = CaptureEngine()
    serving.register_engine("chatterbox", default_engine, default=True)
    serving.register_engine("atom_vllm", atom_engine)

    request = AudioSpeechRequest(
        input="hello",
        backend="atom_vllm",
        voice=str(DEFAULT_US_FEMALE_VOICE),
        batch_size=4,
        chunk_chars=160,
        cfg_weight=0.5,
        diffusion_steps=3,
        temperature=0.75,
        top_p=0.95,
        min_p=0.01,
    )
    model_name, engine = serving._resolve_engine(request.model)
    model_name, engine = serving._resolve_backend_engine(model_name, engine, request.backend)
    assert model_name == "atom_vllm"

    serving._run_engine(engine, request)

    assert atom_engine.kwargs["ref_audio_path"] == str(DEFAULT_US_FEMALE_VOICE)
    assert atom_engine.kwargs["batch_size"] == 4
    assert atom_engine.kwargs["cfg_weight"] == 0.5
    assert atom_engine.kwargs["diffusion_steps"] == 3
    assert atom_engine.kwargs["top_p"] == 0.95


def test_extra_params_backend_is_final_route_override():
    serving = SpeechServing()
    transformer_engine = CaptureEngine()
    atom_engine = CaptureEngine()
    serving.register_engine("chatterbox_transformer", transformer_engine, default=True)
    serving.register_engine("atom_vllm", atom_engine)

    request = AudioSpeechRequest(
        input="hello",
        backend="transformer",
        extra_params={"backend": "atom_vllm", "diffusion_steps": 2},
    )

    model_name, engine = serving._resolve_engine(request.model)
    model_name, engine = serving._resolve_backend_engine(
        model_name,
        engine,
        serving._request_backend(request),
    )
    serving._run_engine(engine, request)

    assert model_name == "atom_vllm"
    assert atom_engine.kwargs["backend"] == "atom_vllm"
    assert atom_engine.kwargs["diffusion_steps"] == 2


def test_generate_kwargs_filter_drops_unsupported_backend_args():
    class NarrowEngine:
        def generate(self, text, exaggeration):
            return np.zeros(1, dtype=np.float32), {}

    serving = SpeechServing()
    request = AudioSpeechRequest(
        input="hello",
        backend="atom_vllm",
        batch_size=4,
        exaggeration=0.7,
    )

    kwargs = serving._filter_generate_kwargs(
        NarrowEngine(),
        serving._speech_generate_kwargs(request),
    )

    assert kwargs == {"exaggeration": 0.7}


def test_split_text_prefers_sentence_boundaries():
    chunks = _split_text("One sentence. Two sentence. Three sentence.", 20)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 25 for chunk in chunks)


@pytest.mark.asyncio
async def test_batch_speech_uses_backend_generate_batch():
    serving = SpeechServing()
    engine = CaptureBatchEngine()
    serving.register_engine("atom_vllm", engine, default=True)

    response = await serving.create_speech_batch(
        BatchSpeechRequest(
            model="atom_vllm",
            backend="atom_vllm",
            batch_size=4,
            chunk_chars=160,
            items=[
                SpeechBatchItem(input="one"),
                SpeechBatchItem(input="two"),
            ],
        )
    )

    assert response.succeeded == 2
    assert engine.batch_texts == ["one", "two"]
    assert engine.kwargs["batch_size"] == 4
    assert engine.kwargs["chunk_chars"] == 160
