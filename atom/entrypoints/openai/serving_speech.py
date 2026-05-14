# SPDX-License-Identifier: Apache-2.0
"""ATOM multi-model TTS serving — OpenAI /v1/audio/speech compatible.

Supports Chatterbox (standard + turbo), Fish Speech S2 Pro, VoxCPM2,
and future models via a pluggable engine registry.

Ported from vLLM-Omni's OmniOpenAIServingSpeech, adapted for ATOM's
standalone engine architecture (no stage pipeline).
"""

import asyncio
import base64
import io
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from atom.audio.protocol import (
    AudioSpeechRequest,
    BatchSpeechRequest,
    BatchSpeechResponse,
    SpeechBatchItemResult,
    VoiceDeleteResponse,
    VoiceInfo,
    VoiceListResponse,
    VoiceUploadResponse,
)
from atom.audio.utils import apply_speed_adjustment, audio_to_bytes, create_wav_header

logger = logging.getLogger("atom.audio")

_REF_AUDIO_MIN_DURATION = 1.0  # seconds
_REF_AUDIO_MAX_DURATION = 30.0  # seconds


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    filename = os.path.basename(filename)
    sanitized = re.sub(r"[^a-zA-Z0-9_.\-]", "_", filename)
    if not sanitized:
        sanitized = "file"
    return sanitized[:255]


def _validate_speaker_name(name: str) -> str:
    """Trim and reject empty / path-separator / NUL / reserved voice names."""
    trimmed = (name or "").strip()
    if not trimmed or trimmed in (".", "..") or any(c in trimmed for c in "/\\\x00"):
        raise ValueError(
            f"Invalid voice name {name!r}: must be non-empty, no path separators or NUL"
        )
    return trimmed


def _validate_path_within_directory(file_path: Path, directory: Path) -> bool:
    """Validate that file_path is within the specified directory."""
    try:
        file_path_resolved = file_path.resolve()
        directory_resolved = directory.resolve()
        return directory_resolved in file_path_resolved.parents or directory_resolved == file_path_resolved
    except Exception:
        return False


class TTSEngine:
    """Base interface for TTS engines."""

    def generate(
        self,
        text: str,
        ref_audio_path: str | None = None,
        ref_audio_array: np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict]:
        """Generate speech audio from text.

        Returns (wav_array, metrics_dict).
        """
        raise NotImplementedError

    @property
    def sample_rate(self) -> int:
        raise NotImplementedError

    @property
    def model_type(self) -> str:
        raise NotImplementedError


class SpeechServing:
    """Multi-model TTS serving with voice management.

    Manages multiple TTS engines and routes requests based on model name.
    Supports voice upload/delete, batch synthesis, and PCM streaming.
    """

    def __init__(self) -> None:
        self._engines: dict[str, Any] = {}  # model_name -> engine
        self._default_engine: str | None = None
        self._upload_lock = asyncio.Lock()

        # Voice management
        speaker_dir = os.environ.get(
            "SPEAKER_SAMPLES_DIR",
            os.path.expanduser("~/.cache/atom/speakers"),
        )
        self.uploaded_speakers_dir = Path(speaker_dir).expanduser()
        self.uploaded_speakers_dir.mkdir(parents=True, exist_ok=True)
        self.uploaded_speakers: dict[str, dict] = {}  # voice_name_lower -> info
        self.supported_speakers: set[str] = {"default"}
        self._ref_audio_data_url_cache: dict[str, str] = {}
        self._max_uploaded_speakers = int(
            os.environ.get("SPEAKER_MAX_UPLOADED", "1000")
        )
        self._last_upload_ts = 0
        self._restore_uploaded_speakers()

    def register_engine(
        self,
        name: str,
        engine: Any,
        *,
        default: bool = False,
    ) -> None:
        """Register a TTS engine under a model name."""
        self._engines[name] = engine
        if default or self._default_engine is None:
            self._default_engine = name
        logger.info("Registered TTS engine: %s (default=%s)", name, default)

    def _resolve_engine(self, model: str | None) -> tuple[str, Any]:
        """Resolve model name to engine. Returns (name, engine)."""
        if not self._engines:
            raise HTTPException(status_code=503, detail="No TTS engines loaded")

        if model is not None:
            # Try exact match
            if model in self._engines:
                return model, self._engines[model]
            # Try case-insensitive / substring match
            model_lower = model.lower()
            for name, engine in self._engines.items():
                if model_lower in name.lower():
                    return name, engine
            raise HTTPException(
                status_code=400,
                detail=f"Unknown TTS model '{model}'. Available: {list(self._engines.keys())}",
            )

        if self._default_engine is None:
            raise HTTPException(status_code=503, detail="No default TTS engine")
        return self._default_engine, self._engines[self._default_engine]

    # ------------------------------------------------------------------
    #  Core speech generation
    # ------------------------------------------------------------------

    async def create_speech(self, request: AudioSpeechRequest) -> Response:
        """Generate speech from text and return audio bytes."""
        t0 = time.time()
        model_name, engine = self._resolve_engine(request.model)

        # Resolve uploaded voice to ref_audio
        self._apply_uploaded_speaker(request)

        try:
            wav, metrics = self._run_engine(engine, request)
        except Exception as e:
            logger.error("Speech generation failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Speech generation failed: {e}")

        # Apply speed adjustment if needed
        speed = request.speed or 1.0
        sample_rate = self._get_sample_rate(engine)
        if speed != 1.0:
            wav, sample_rate = apply_speed_adjustment(wav, speed, sample_rate)

        # Streaming PCM response
        if request.stream and request.response_format == "pcm":
            return self._stream_pcm(wav, sample_rate)

        # Streaming WAV response
        if request.stream and request.response_format == "wav":
            return self._stream_wav(wav, sample_rate)

        audio_bytes, media_type = audio_to_bytes(
            wav, sample_rate, request.response_format,
        )

        elapsed = time.time() - t0
        logger.info(
            "Speech [%s]: %.1fs audio in %.1fs (%.1f tok/s, RTF=%.2f) fmt=%s",
            model_name,
            metrics.get("audio_duration", 0),
            elapsed,
            metrics.get("tok_per_sec", 0),
            metrics.get("rtf", 0),
            request.response_format,
        )

        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "X-Audio-Duration": f"{metrics.get('audio_duration', 0):.2f}",
                "X-Generation-Tokens": str(metrics.get("num_tokens", 0)),
                "X-Tokens-Per-Second": f"{metrics.get('tok_per_sec', 0):.1f}",
                "X-Realtime-Factor": f"{metrics.get('rtf', 0):.2f}",
                "X-Model": model_name,
            },
        )

    def _run_engine(self, engine: Any, request: AudioSpeechRequest) -> tuple[np.ndarray, dict]:
        """Run the appropriate engine based on its type."""
        from atom.audio.chatterbox.engine import ChatterboxEngine

        if isinstance(engine, ChatterboxEngine):
            return engine.generate(
                text=request.input,
                ref_audio_path=request.ref_audio or (
                    request.voice
                    if request.voice and request.voice != "default"
                    else None
                ),
                exaggeration=request.exaggeration,
                max_tokens=request.max_tokens,
                repetition_penalty=request.repetition_penalty,
                seed=request.seed,
            )

        # Generic engine interface
        kwargs: dict[str, Any] = {}
        if request.ref_audio is not None:
            kwargs["ref_audio_path"] = request.ref_audio
        if request.ref_text is not None:
            kwargs["ref_text"] = request.ref_text
        if request.max_new_tokens is not None:
            kwargs["max_tokens"] = request.max_new_tokens
        elif request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.voice and request.voice != "default":
            kwargs["voice"] = request.voice
        if request.extra_params:
            kwargs.update(request.extra_params)

        return engine.generate(text=request.input, **kwargs)

    def _get_sample_rate(self, engine: Any) -> int:
        """Get sample rate from engine."""
        if hasattr(engine, "sample_rate"):
            return engine.sample_rate
        if hasattr(engine, "service") and hasattr(engine.service, "sample_rate"):
            return engine.service.sample_rate
        return 24000  # Default

    def _stream_pcm(self, wav: np.ndarray, sample_rate: int) -> StreamingResponse:
        """Stream raw PCM audio."""
        pcm = (wav * 32767).clip(-32768, 32767).astype(np.int16)
        chunk_size = sample_rate  # 1 second chunks

        async def pcm_generator():
            for start in range(0, len(pcm), chunk_size):
                yield pcm[start : start + chunk_size].tobytes()

        return StreamingResponse(
            pcm_generator(),
            media_type="audio/pcm",
            headers={"X-Sample-Rate": str(sample_rate)},
        )

    def _stream_wav(self, wav: np.ndarray, sample_rate: int) -> StreamingResponse:
        """Stream WAV with header + PCM chunks."""
        pcm = (wav * 32767).clip(-32768, 32767).astype(np.int16)
        chunk_size = sample_rate  # 1 second chunks

        async def wav_generator():
            yield create_wav_header(sample_rate)
            for start in range(0, len(pcm), chunk_size):
                yield pcm[start : start + chunk_size].tobytes()

        return StreamingResponse(
            wav_generator(),
            media_type="audio/wav",
            headers={"X-Sample-Rate": str(sample_rate)},
        )

    # ------------------------------------------------------------------
    #  Batch synthesis
    # ------------------------------------------------------------------

    async def create_speech_batch(
        self,
        request: BatchSpeechRequest,
    ) -> BatchSpeechResponse:
        """Batch speech synthesis."""
        results = []
        succeeded = 0
        failed = 0
        _, engine = self._resolve_engine(request.model)

        for idx, item in enumerate(request.items):
            try:
                item_request = AudioSpeechRequest(
                    input=item.input,
                    model=request.model,
                    voice=item.voice or request.voice,
                    response_format=item.response_format or request.response_format,
                    ref_audio=item.ref_audio or request.ref_audio,
                    ref_text=item.ref_text or request.ref_text,
                    exaggeration=(
                        item.exaggeration
                        if item.exaggeration is not None
                        else request.exaggeration
                    ),
                    max_tokens=(
                        item.max_tokens
                        if item.max_tokens is not None
                        else request.max_tokens
                    ),
                    repetition_penalty=request.repetition_penalty,
                )
                wav, metrics = self._run_engine(engine, item_request)
                sample_rate = self._get_sample_rate(engine)
                audio_bytes, media_type = audio_to_bytes(
                    wav, sample_rate,
                    item.response_format or request.response_format,
                )
                results.append(SpeechBatchItemResult(
                    index=idx,
                    status="success",
                    audio_data=base64.b64encode(audio_bytes).decode(),
                    media_type=media_type,
                ))
                succeeded += 1
            except Exception as e:
                results.append(SpeechBatchItemResult(
                    index=idx,
                    status="error",
                    error=str(e),
                ))
                failed += 1

        import uuid
        return BatchSpeechResponse(
            id=f"batch-{uuid.uuid4().hex[:12]}",
            results=results,
            total=len(request.items),
            succeeded=succeeded,
            failed=failed,
        )

    # ------------------------------------------------------------------
    #  Voice management
    # ------------------------------------------------------------------

    async def list_voices(self) -> VoiceListResponse:
        """List available voices."""
        voices = []

        # Built-in voices
        for name in sorted(self.supported_speakers - set(self.uploaded_speakers.keys())):
            voices.append(VoiceInfo(name=name, description="Built-in voice"))

        # Uploaded voices
        for name, info in sorted(self.uploaded_speakers.items()):
            voices.append(VoiceInfo(
                name=info.get("name", name),
                description=info.get("speaker_description"),
                model_type=info.get("embedding_source", "audio"),
                sample_rate=info.get("sample_rate"),
                created_at=info.get("created_at"),
                file_size=info.get("file_size"),
            ))

        # Engine-specific voices
        for eng_name, engine in self._engines.items():
            if hasattr(engine, "list_voices"):
                for v in engine.list_voices():
                    voices.append(VoiceInfo(
                        name=v,
                        description=f"Built-in ({eng_name})",
                        model_type=eng_name,
                    ))

        return VoiceListResponse(voices=voices)

    async def upload_voice(
        self,
        audio_file: UploadFile,
        consent: str,
        name: str,
        *,
        ref_text: str | None = None,
        speaker_description: str | None = None,
    ) -> dict:
        """Upload a new voice sample."""
        name = _validate_speaker_name(name)
        if ref_text is not None:
            ref_text = ref_text.strip() or None
        if speaker_description is not None:
            speaker_description = speaker_description.strip() or None

        # Validate file size (max 10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024
        audio_file.file.seek(0, 2)
        file_size = audio_file.file.tell()
        audio_file.file.seek(0)

        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum limit of 10MB. Got {file_size} bytes.")

        # Validate MIME type
        mime_type = audio_file.content_type
        if mime_type == "application/octet-stream":
            ext_map = {
                ".wav": "audio/wav", ".mp3": "audio/mpeg", ".mpeg": "audio/mpeg",
                ".flac": "audio/flac", ".ogg": "audio/ogg", ".aac": "audio/aac",
                ".webm": "audio/webm", ".mp4": "audio/mp4",
            }
            for ext, mt in ext_map.items():
                if audio_file.filename and audio_file.filename.lower().endswith(ext):
                    mime_type = mt
                    break
            else:
                mime_type = "audio/wav"

        allowed_types = {
            "audio/mpeg", "audio/wav", "audio/x-wav", "audio/ogg",
            "audio/aac", "audio/flac", "audio/webm", "audio/mp4",
        }
        if mime_type not in allowed_types:
            raise ValueError(f"Unsupported MIME type: {mime_type}. Allowed: {allowed_types}")

        content = await audio_file.read()

        async with self._upload_lock:
            voice_name_lower = name.lower()
            self._evict_existing_upload(voice_name_lower, name)
            self._check_upload_cap()

            sanitized_name = _sanitize_filename(name)
            sanitized_consent = _sanitize_filename(consent)
            timestamp = self._next_upload_timestamp()

            # Decode and validate audio
            try:
                wav_np, sr = sf.read(io.BytesIO(content))
            except Exception as e:
                raise ValueError(f"Could not decode audio file: {e}")

            duration = len(wav_np) / sr if sr > 0 else 0.0
            if duration < _REF_AUDIO_MIN_DURATION:
                raise ValueError(
                    f"Reference audio too short ({duration:.1f}s). "
                    f"At least {_REF_AUDIO_MIN_DURATION:.0f}s required."
                )
            if duration > _REF_AUDIO_MAX_DURATION:
                raise ValueError(
                    f"Reference audio too long ({duration:.1f}s). "
                    f"Maximum {_REF_AUDIO_MAX_DURATION:.0f}s supported."
                )

            # Save as safetensors if available, otherwise WAV
            filename = f"{sanitized_name}_{sanitized_consent}_{timestamp}"
            speaker_data: dict[str, Any] = {
                "name": name,
                "voice_name_lower": voice_name_lower,
                "consent": consent,
                "created_at": timestamp,
                "mime_type": mime_type,
                "original_filename": audio_file.filename or "upload.wav",
                "file_size": file_size,
                "sample_rate": int(sr),
                "ref_text": ref_text,
                "embedding_source": "audio",
            }
            if speaker_description:
                speaker_data["speaker_description"] = speaker_description

            try:
                import torch
                from safetensors.torch import save_file
                file_path = self.uploaded_speakers_dir / f"{filename}.safetensors"
                if not _validate_path_within_directory(file_path, self.uploaded_speakers_dir):
                    raise ValueError("Invalid file path: potential path traversal")
                audio_tensor = torch.from_numpy(
                    np.asarray(wav_np, dtype=np.float32)
                ).contiguous()
                metadata = {k: str(v) for k, v in speaker_data.items() if v is not None and k != "file_path"}
                save_file({"audio": audio_tensor}, str(file_path), metadata=metadata)
            except ImportError:
                # Fallback: save as WAV
                file_path = self.uploaded_speakers_dir / f"{filename}.wav"
                if not _validate_path_within_directory(file_path, self.uploaded_speakers_dir):
                    raise ValueError("Invalid file path: potential path traversal")
                sf.write(str(file_path), wav_np, sr, format="WAV")

            speaker_data["file_path"] = str(file_path)
            self.uploaded_speakers[voice_name_lower] = speaker_data
            self.supported_speakers.add(voice_name_lower)

        logger.info("Uploaded voice '%s' (%.1fs, %dHz)", name, duration, sr)

        result = {
            "name": name,
            "consent": consent,
            "created_at": timestamp,
            "mime_type": mime_type,
            "file_size": file_size,
            "sample_rate": int(sr),
            "duration": round(duration, 2),
        }
        if ref_text:
            result["ref_text"] = ref_text
        if speaker_description:
            result["speaker_description"] = speaker_description
        return result

    async def delete_voice(self, name: str) -> bool:
        """Delete an uploaded voice. Returns True if successful."""
        async with self._upload_lock:
            voice_name_lower = name.lower()
            if voice_name_lower not in self.uploaded_speakers:
                return False

            speaker_info = self.uploaded_speakers.pop(voice_name_lower)
            self.supported_speakers.discard(voice_name_lower)
            self._ref_audio_data_url_cache.pop(voice_name_lower, None)

            file_path = speaker_info.get("file_path")
            if file_path:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning("Failed to delete file for '%s': %s", name, e)

        logger.info("Deleted voice '%s'", name)
        return True

    # ------------------------------------------------------------------
    #  Voice resolution helpers
    # ------------------------------------------------------------------

    def _apply_uploaded_speaker(self, request: AudioSpeechRequest) -> None:
        """Resolve request.voice against uploaded speakers, mutating ref_audio."""
        if request.voice is None or request.ref_audio is not None:
            return

        voice_lower = request.voice.lower()
        if voice_lower not in self.uploaded_speakers:
            return

        audio_data = self._get_uploaded_audio_data(request.voice)
        if audio_data:
            request.ref_audio = audio_data
            if not request.ref_text:
                stored_ref_text = self.uploaded_speakers[voice_lower].get("ref_text")
                if stored_ref_text:
                    request.ref_text = stored_ref_text
            logger.info("Resolved uploaded voice '%s'", voice_lower)

    def _get_uploaded_audio_data(self, voice_name: str) -> str | None:
        """Return a base64-encoded WAV data URL for an uploaded voice."""
        voice_name_lower = voice_name.lower()
        cached = self._ref_audio_data_url_cache.get(voice_name_lower)
        if cached is not None:
            return cached

        info = self.uploaded_speakers.get(voice_name_lower)
        if info is None:
            return None

        file_path = Path(info["file_path"])
        if not file_path.exists():
            return None

        try:
            if str(file_path).endswith(".safetensors"):
                from safetensors import safe_open
                with safe_open(str(file_path), framework="pt") as f:
                    if "audio" not in f.keys():
                        return None
                    samples = f.get_tensor("audio").numpy()
                    sr = int((f.metadata() or {}).get("sample_rate", info.get("sample_rate", 24000)))
            else:
                samples, sr = sf.read(str(file_path))

            buf = io.BytesIO()
            sf.write(buf, samples, sr, format="WAV")
            audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_url = f"data:audio/wav;base64,{audio_b64}"
        except Exception as e:
            logger.error("Could not encode voice %s as WAV: %s", voice_name, e)
            return None

        self._ref_audio_data_url_cache[voice_name_lower] = data_url
        return data_url

    def _next_upload_timestamp(self) -> int:
        ts = max(int(time.time()), self._last_upload_ts + 1)
        self._last_upload_ts = ts
        return ts

    def _check_upload_cap(self) -> None:
        if len(self.uploaded_speakers) >= self._max_uploaded_speakers:
            raise ValueError(
                f"Uploaded voice limit reached ({self._max_uploaded_speakers}). "
                f"Delete an existing voice first."
            )

    def _evict_existing_upload(self, voice_name_lower: str, name: str) -> None:
        if voice_name_lower not in self.uploaded_speakers:
            return
        old = self.uploaded_speakers.pop(voice_name_lower)
        self.supported_speakers.discard(voice_name_lower)
        self._ref_audio_data_url_cache.pop(voice_name_lower, None)
        old_path = old.get("file_path")
        if old_path:
            try:
                Path(old_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to remove previous file for '%s': %s", name, e)

    def _restore_uploaded_speakers(self) -> None:
        """Scan uploaded_speakers_dir and rebuild state."""
        restored = 0

        # Restore from safetensors files
        try:
            from safetensors import safe_open
            for path in sorted(self.uploaded_speakers_dir.glob("*.safetensors")):
                try:
                    with safe_open(str(path), framework="pt") as f:
                        header = dict(f.metadata() or {})
                except Exception as e:
                    logger.warning("Could not read voice file %s: %s", path, e)
                    continue
                voice_name_lower = header.get("voice_name_lower") or header.get("name", "").lower()
                if not voice_name_lower:
                    continue
                speaker_data = dict(header)
                speaker_data["file_path"] = str(path)
                speaker_data.setdefault("name", voice_name_lower)
                for k in ("created_at", "file_size", "sample_rate"):
                    if k in speaker_data:
                        try:
                            speaker_data[k] = int(speaker_data[k])
                        except ValueError:
                            pass
                self.uploaded_speakers[voice_name_lower] = speaker_data
                self.supported_speakers.add(voice_name_lower)
                self._last_upload_ts = max(
                    self._last_upload_ts, int(speaker_data.get("created_at", 0))
                )
                restored += 1
        except ImportError:
            pass

        # Restore from WAV files
        for path in sorted(self.uploaded_speakers_dir.glob("*.wav")):
            name = path.stem
            voice_name_lower = name.lower()
            if voice_name_lower in self.uploaded_speakers:
                continue
            self.uploaded_speakers[voice_name_lower] = {
                "name": name,
                "voice_name_lower": voice_name_lower,
                "file_path": str(path),
                "embedding_source": "audio",
            }
            self.supported_speakers.add(voice_name_lower)
            restored += 1

        if restored:
            logger.info(
                "Restored %d uploaded voice(s) from %s",
                restored, self.uploaded_speakers_dir,
            )
