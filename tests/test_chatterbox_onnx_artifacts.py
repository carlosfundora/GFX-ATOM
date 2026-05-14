import sys
import types
from pathlib import Path

import atom.audio as audio_api
import pytest

from atom.audio.chatterbox import onnx_artifacts
from atom.audio.chatterbox.engine import ChatterboxEngine
from atom.audio.chatterbox.service import ChatterboxService
from atom.audio.runtime import OnnxCpuRuntimeConfig, create_cpu_session_options


def test_q8_component_resolution_prefers_sidecar(tmp_path: Path) -> None:
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    q8_path = onnx_dir / "speech_encoder_q8.onnx"
    fallback_path = onnx_dir / "speech_encoder.onnx"
    q8_path.write_text("q8", encoding="utf-8")
    fallback_path.write_text("fp", encoding="utf-8")

    resolved = onnx_artifacts.resolve_component_path(onnx_dir, "speech_encoder", "q8")

    assert resolved == q8_path


def test_q8_component_resolution_falls_back_to_fp_component(tmp_path: Path) -> None:
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    fallback_path = onnx_dir / "conditional_decoder.onnx"
    fallback_path.write_text("fp", encoding="utf-8")

    resolved = onnx_artifacts.resolve_component_path(onnx_dir, "conditional_decoder", "q8")

    assert resolved == fallback_path


def test_language_model_q8_resolution_falls_back_to_fp16(tmp_path: Path) -> None:
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    fallback_path = onnx_dir / "language_model_fp16.onnx"
    fallback_path.write_text("fp16", encoding="utf-8")

    resolved = onnx_artifacts.resolve_component_path(onnx_dir, "language_model", "q8")

    assert resolved == fallback_path


def test_create_cpu_session_options_uses_requested_thread_shape() -> None:
    opts = create_cpu_session_options(num_threads=12)

    assert opts.intra_op_num_threads == 12
    assert opts.inter_op_num_threads == 1


def test_audio_runtime_config_is_public_audio_api() -> None:
    config = OnnxCpuRuntimeConfig(num_threads=6, inter_op_threads=1)

    assert audio_api.OnnxCpuRuntimeConfig is OnnxCpuRuntimeConfig
    assert audio_api.create_cpu_session_options is create_cpu_session_options
    assert config.intra_op_threads == 6
    assert config.create_session_options().intra_op_num_threads == 6


def test_chatterbox_default_reference_conditioning_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ChatterboxService(model_dir="/tmp/nonexistent")
    calls = []

    def fake_encode_reference(*, audio_path=None, audio_array=None):
        calls.append((audio_path, audio_array))
        return {"call": len(calls)}

    monkeypatch.setattr(service, "encode_reference", fake_encode_reference)

    first = service.get_reference_data()
    second = service.get_reference_data()
    explicit = service.get_reference_data(audio_path="/tmp/voice.wav")

    assert first is second
    assert first == {"call": 1}
    assert explicit == {"call": 2}
    assert calls == [(None, None), ("/tmp/voice.wav", None)]


def test_embed_single_token_preserves_requested_exaggeration(monkeypatch) -> None:
    service = ChatterboxService(model_dir="/tmp/nonexistent")
    captured = {}

    class FakeInput:
        def __init__(self, name):
            self.name = name

    class FakeEmbedSession:
        def get_inputs(self):
            return [FakeInput("input_ids"), FakeInput("exaggeration")]

        def run(self, _outputs, inputs):
            captured.update(inputs)
            return [[[0.0]]]

    service._embed_tokens = FakeEmbedSession()

    service.embed_single_token([[123]], exaggeration=0.8)

    assert captured["exaggeration"].tolist() == pytest.approx([0.8])


def test_quantize_dynamic_q8_wires_ort_quantizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {}
    fake_onnx = types.ModuleType("onnx")
    fake_quantization = types.ModuleType("onnxruntime.quantization")

    class FakeQuantType:
        QInt8 = object()

    def fake_quantize_dynamic(**kwargs):
        calls.update(kwargs)
        Path(kwargs["model_output"]).write_text("q8", encoding="utf-8")

    fake_quantization.QuantType = FakeQuantType
    fake_quantization.quantize_dynamic = fake_quantize_dynamic
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", fake_quantization)

    source = tmp_path / "model.onnx"
    output = tmp_path / "model_q8.onnx"
    source.write_text("fp", encoding="utf-8")

    result = onnx_artifacts.quantize_dynamic_q8(source, output)

    assert result.quantized is True
    assert calls["model_input"] == source
    assert calls["model_output"] == output
    assert calls["weight_type"] is FakeQuantType.QInt8
    assert calls["use_external_data_format"] is True


def test_quantize_dynamic_q8_reports_missing_onnx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "onnx", raising=False)

    source = tmp_path / "model.onnx"
    output = tmp_path / "model_q8.onnx"
    source.write_text("fp", encoding="utf-8")

    with pytest.raises(RuntimeError, match="requires the optional `onnx` package"):
        onnx_artifacts.quantize_dynamic_q8(source, output)


def test_engine_onnx_cpu_loader_uses_q8_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True)
    q8_lm = onnx_dir / "language_model_q8.onnx"
    q8_lm.write_text("q8", encoding="utf-8")

    loaded = {}

    def fake_create_cpu_session(path: str, **kwargs):
        loaded["path"] = path
        loaded["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        "atom.audio.chatterbox.engine.create_cpu_inference_session",
        fake_create_cpu_session,
    )

    engine = ChatterboxEngine(
        model_dir=str(model_dir),
        backbone_dir=None,
        onnx_variant="q8",
        num_threads=3,
    )
    engine._load_onnx_lm()

    assert loaded["path"] == str(q8_lm)
    assert loaded["kwargs"]["num_threads"] == 3
    assert engine._use_gpu_backbone is False
