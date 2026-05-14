# SPDX-License-Identifier: Apache-2.0
"""LFM2.5-Audio bridge for ATOM's audio API.

This module treats the local audio-enabled llama.cpp engine as the runtime
boundary.  The donor tree owns the C++/GGUF details; ATOM owns request shaping,
SSE parsing, optional process launch, and OpenAI-compatible audio helpers.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
import numpy as np
import soundfile as sf

logger = logging.getLogger("atom.audio.lfm25")

DEFAULT_ENGINE_DIR = Path("/home/local/ai/engines/llama.cpp-audio-max")
DEFAULT_BINARY = DEFAULT_ENGINE_DIR / "bin" / "llama-liquid-audio-server"
DEFAULT_MODEL_DIRS = (
    Path("/home/local/ai/models/audio/LiquidAI/LFM2.5-Audio-1.5B"),
    Path("/home/local/ai/audio/models/LiquidAI/LFM2.5-Audio-1.5B"),
    Path("/home/local/ai/projects/DEMERZEL/models/lfm2.5-audio-1.5b"),
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30008
DEFAULT_SAMPLE_RATE = 24000

_SYSTEM_PROMPTS = {
    "asr": (
        "Transcribe the user's speech to text verbatim. "
        "Output only the transcript, no commentary."
    ),
    "tts_us_female": "Perform TTS. Use the US female voice.",
    "interleaved": (
        "Your name is Demrezel. You are NOT Lili, you are NOT Liquid Lili, "
        "and you must never identify as anyone other than Demrezel. "
        "Be concise, useful, and conversational. "
        "Respond with interleaved text and audio."
    ),
}


@dataclass(frozen=True)
class LFM25AudioPaths:
    model: Path
    mmproj: Path
    vocoder: Path
    tokenizer: Path


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_lfm25_audio_paths(
    model_dir: str | os.PathLike[str] | None = None,
    *,
    precision: str = "q8",
) -> LFM25AudioPaths:
    """Resolve local LFM2.5-Audio GGUF artifacts.

    ``precision`` accepts ``q8``/``quantized`` and ``f16``/``unquantized``.
    The multimodal projector has historically been available as F16 in
    DEMERZEL even when the main model is Q8, so that component falls back
    across compatible names.
    """

    precision_key = precision.lower().replace("_", "").replace("-", "")
    prefer_f16 = precision_key in {"f16", "fp16", "float16", "unquantized"}
    dirs = [Path(model_dir).expanduser()] if model_dir else list(DEFAULT_MODEL_DIRS)
    root = _first_existing(dirs)
    if root is None:
        searched = ", ".join(str(item) for item in dirs)
        raise FileNotFoundError(f"LFM2.5-Audio model directory not found; searched {searched}")

    model_names = (
        ["LFM2.5-Audio-1.5B-F16.gguf", "LFM2.5-Audio-1.5B-Q8_0.gguf"]
        if prefer_f16
        else ["LFM2.5-Audio-1.5B-Q8_0.gguf", "LFM2.5-Audio-1.5B-F16.gguf"]
    )
    mmproj_names = (
        ["mmproj-LFM2.5-Audio-1.5B-F16.gguf", "mmproj-LFM2.5-Audio-1.5B-Q8_0.gguf"]
        if prefer_f16
        else ["mmproj-LFM2.5-Audio-1.5B-Q8_0.gguf", "mmproj-LFM2.5-Audio-1.5B-F16.gguf"]
    )
    vocoder_names = (
        "vocoder-LFM2.5-Audio-1.5B-Q8_0.gguf",
        "vocoder-LFM2.5-Audio-1.5B-F16.gguf",
    )
    tokenizer_names = (
        "tokenizer-LFM2.5-Audio-1.5B-Q8_0.gguf",
        "tokenizer-LFM2.5-Audio-1.5B-F16.gguf",
    )

    model = _first_existing(root / name for name in model_names)
    mmproj = _first_existing(root / name for name in mmproj_names)
    vocoder = _first_existing(root / name for name in vocoder_names)
    tokenizer = _first_existing(root / name for name in tokenizer_names)
    missing = [
        name
        for name, value in (
            ("model", model),
            ("mmproj", mmproj),
            ("vocoder", vocoder),
            ("tokenizer", tokenizer),
        )
        if value is None
    ]
    if missing:
        raise FileNotFoundError(
            f"LFM2.5-Audio missing {', '.join(missing)} artifacts under {root}"
        )
    return LFM25AudioPaths(
        model=model,
        mmproj=mmproj,
        vocoder=vocoder,
        tokenizer=tokenizer,
    )


def build_lfm25_server_command(
    paths: LFM25AudioPaths,
    *,
    binary: str | os.PathLike[str] = DEFAULT_BINARY,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    n_gpu_layers: int | str = 0,
    threads: int = 12,
    ctx_size: int = 4096,
    flash_attn: str = "off",
) -> list[str]:
    return [
        str(binary),
        "--model",
        str(paths.model),
        "--mmproj",
        str(paths.mmproj),
        "--model-vocoder",
        str(paths.vocoder),
        "--tts-speaker-file",
        str(paths.tokenizer),
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(ctx_size),
        "--threads",
        str(threads),
        "--threads-batch",
        str(threads),
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--flash-attn",
        flash_attn,
        "--parallel",
        "1",
        "--gpu-layers",
        str(n_gpu_layers),
    ]


def lfm25_runtime_env(engine_dir: Path = DEFAULT_ENGINE_DIR) -> dict[str, str]:
    env = os.environ.copy()
    lib_path = str(engine_dir / "bin")
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    env.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    env.setdefault("GGML_HIP_DEVICE", "0")
    env.setdefault("GGML_HIP_UMM", "0")
    return env


def _pcm_chunks_to_wav_bytes(chunks: list[str], sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    pcm = b"".join(base64.b64decode(chunk) for chunk in chunks if chunk)
    if not pcm:
        return b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _wav_bytes_to_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


class LFM25AudioClient:
    """Client for a running ``llama-liquid-audio-server`` instance."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        model: str = "lfm2.5-audio",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or f"http://{host}:{port}/v1").rstrip("/")
        self.model = model
        self.timeout = timeout

    def _chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        text_parts: list[str] = []
        audio_chunks: list[str] = []
        sample_rate = DEFAULT_SAMPLE_RATE
        request_payload = {
            "model": self.model,
            "stream": True,
            "reset_context": True,
            **payload,
        }

        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=request_payload,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if event.get("error"):
                    error = event["error"]
                    raise RuntimeError(error.get("message") or str(error))
                delta = ((event.get("choices") or [{}])[0].get("delta") or {})
                if delta.get("content"):
                    text_parts.append(delta["content"])
                audio = delta.get("audio") or {}
                if audio.get("data"):
                    audio_chunks.append(audio["data"])
                    sample_rate = int(audio.get("sample_rate") or sample_rate)

        wav_bytes = _pcm_chunks_to_wav_bytes(audio_chunks, sample_rate)
        return {
            "text": "".join(text_parts).strip(),
            "wav_bytes": wav_bytes,
            "sample_rate": sample_rate,
            "audio_chunks": len(audio_chunks),
        }

    def text_to_speech(self, text: str, *, max_tokens: int = 512) -> dict[str, Any]:
        return self._chat_completion(
            {
                "modalities": ["audio"],
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPTS["tts_us_female"]},
                    {"role": "user", "content": text},
                ],
            }
        )

    def transcribe_wav_bytes(
        self,
        wav_bytes: bytes,
        *,
        language: str = "auto",
        max_tokens: int = 512,
    ) -> str:
        encoded = base64.b64encode(wav_bytes).decode("ascii")
        system_prompt = _SYSTEM_PROMPTS["asr"]
        if language and language != "auto":
            system_prompt += f" Source language: {language}."
        result = self._chat_completion(
            {
                "modalities": ["text"],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": encoded, "format": "wav"},
                            }
                        ],
                    },
                ],
            }
        )
        return result["text"]

    def speech_to_speech(
        self,
        wav_bytes: bytes,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        encoded = base64.b64encode(wav_bytes).decode("ascii")
        return self._chat_completion(
            {
                "modalities": ["text", "audio"],
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt or _SYSTEM_PROMPTS["interleaved"],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": encoded, "format": "wav"},
                            }
                        ],
                    },
                ],
            }
        )


class LFM25AudioEngine:
    """Speech API engine that proxies TTS to LFM2.5-Audio."""

    def __init__(self, client: LFM25AudioClient | None = None) -> None:
        self.client = client or LFM25AudioClient()
        self._sample_rate = DEFAULT_SAMPLE_RATE

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def model_type(self) -> str:
        return "lfm2.5-audio"

    def generate(self, text: str, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        t0 = time.time()
        max_tokens = int(kwargs.get("max_tokens") or kwargs.get("max_new_tokens") or 512)
        data = self.client.text_to_speech(text, max_tokens=max_tokens)
        wav_bytes = data.get("wav_bytes") or b""
        if not wav_bytes:
            raise RuntimeError("LFM2.5-Audio returned no audio chunks")
        wav, sample_rate = _wav_bytes_to_float32(wav_bytes)
        self._sample_rate = sample_rate
        total_sec = time.time() - t0
        audio_duration = len(wav) / max(sample_rate, 1)
        metrics = {
            "backend": "lfm2.5-audio",
            "audio_duration": audio_duration,
            "total_sec": total_sec,
            "rtf": total_sec / max(audio_duration, 0.001),
            "num_tokens": 0,
            "tok_per_sec": 0.0,
            "audio_chunks": data.get("audio_chunks", 0),
        }
        return wav, metrics


class ManagedLFM25AudioServer:
    """Optional process wrapper used by benchmark/proof scripts."""

    def __init__(
        self,
        *,
        model_dir: str | os.PathLike[str] | None = None,
        precision: str = "q8",
        port: int = DEFAULT_PORT,
        n_gpu_layers: int | str = 0,
        threads: int = 12,
    ) -> None:
        self.port = port
        self.paths = resolve_lfm25_audio_paths(model_dir, precision=precision)
        self.command = build_lfm25_server_command(
            self.paths,
            port=port,
            n_gpu_layers=n_gpu_layers,
            threads=threads,
        )
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "ManagedLFM25AudioServer":
        self.process = subprocess.Popen(
            self.command,
            env=lfm25_runtime_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
