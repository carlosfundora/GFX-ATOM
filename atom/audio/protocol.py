# SPDX-License-Identifier: Apache-2.0
"""Audio API request/response models — OpenAI /v1/audio/speech compatible.

Ported from vLLM-Omni's protocol/audio.py with multi-model TTS support:
Chatterbox, Fish Speech S2 Pro, VoxCPM2, Qwen3-TTS, and others.
"""

import math
from typing import Any, Literal

import numpy as np
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

_MAX_EMBEDDING_DIM = 8192


# ---------------------------------------------------------------------------
#  Speech synthesis (POST /v1/audio/speech)
# ---------------------------------------------------------------------------


class AudioSpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request with multi-model extensions."""

    input: str = Field(description="Text to synthesize")
    model: str | None = Field(default=None, description="Model identifier")
    voice: str | None = Field(
        default=None,
        validation_alias=AliasChoices("voice", "speaker"),
        description="Speaker/voice to use. Name, path, or 'default'.",
    )
    instructions: str | None = Field(
        default=None,
        description="Instructions for voice style/emotion",
    )
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: Literal["sse", "audio"] | None = "audio"
    stream: bool = Field(
        default=False,
        description="Stream raw PCM audio chunks as they are decoded. "
        "Requires response_format='pcm' or 'wav'.",
    )

    # Voice cloning parameters (shared across models)
    ref_audio: str | None = Field(
        default=None,
        description="Reference audio for voice cloning. URL, base64, or file path.",
    )
    ref_text: str | None = Field(
        default=None,
        description="Transcript of reference audio for voice cloning",
    )

    # Generation parameters
    max_new_tokens: int | None = Field(
        default=None,
        ge=1,
        le=4096,
        description="Maximum tokens to generate",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**63 - 1,
        description="Random seed for reproducible generation",
    )

    # Qwen3-TTS specific
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = Field(
        default=None,
        description="TTS task type: CustomVoice, VoiceDesign, or Base (voice clone)",
    )
    language: str | None = Field(
        default=None,
        description="Language code (e.g., 'Chinese', 'English', 'Auto')",
    )
    x_vector_only_mode: bool | None = Field(
        default=None,
        description="Use speaker embedding only without in-context learning",
    )
    speaker_embedding: list[float] | None = Field(
        default=None,
        max_length=_MAX_EMBEDDING_DIM,
        description="Pre-computed speaker embedding vector. "
        "Mutually exclusive with ref_audio.",
    )
    initial_codec_chunk_frames: int | None = Field(
        default=None,
        ge=0,
        description="Per-request initial chunk size override",
    )

    # Chatterbox-specific
    exaggeration: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Emotion intensity (Chatterbox only)",
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum speech tokens (Chatterbox only)",
    )
    repetition_penalty: float = Field(
        default=1.2,
        ge=1.0,
        le=3.0,
        description="Repetition penalty for speech token generation",
    )

    # Model-specific extra parameters
    extra_params: dict[str, Any] | None = Field(
        default=None,
        description="Optional model-specific parameters",
    )

    @field_validator("stream_format")
    @classmethod
    def validate_stream_format(cls, v: str) -> str:
        if v == "sse":
            raise ValueError("'sse' is not a supported stream_format yet. Use 'audio'.")
        return v

    @field_validator("speaker_embedding")
    @classmethod
    def validate_speaker_embedding(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and not all(math.isfinite(x) for x in v):
            raise ValueError("'speaker_embedding' values must be finite (no NaN or Inf)")
        return v

    @model_validator(mode="after")
    def validate_embedding_constraints(self) -> "AudioSpeechRequest":
        if self.speaker_embedding is not None and self.ref_audio is not None:
            raise ValueError("'speaker_embedding' and 'ref_audio' are mutually exclusive")
        return self

    @model_validator(mode="after")
    def validate_streaming_constraints(self) -> "AudioSpeechRequest":
        if self.stream:
            if self.response_format not in ("pcm", "wav"):
                raise ValueError(
                    "Streaming (stream=true) requires response_format='pcm' or 'wav'. "
                    f"Got response_format='{self.response_format}'."
                )
            if self.speed is None:
                self.speed = 1.0
            elif self.speed != 1.0:
                raise ValueError(
                    "Speed adjustment is not supported when streaming. "
                    "Set speed=1.0 or omit it."
                )
        return self


# ---------------------------------------------------------------------------
#  Batch speech synthesis (POST /v1/audio/speech/batch)
# ---------------------------------------------------------------------------


class SpeechBatchItem(BaseModel):
    """Per-item input for batch speech. Only `input` is required;
    other fields override batch-level defaults when set."""

    input: str
    voice: str | None = None
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] | None = None
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    max_new_tokens: int | None = None
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)
    # Chatterbox-specific
    exaggeration: float | None = None
    max_tokens: int | None = None


class BatchSpeechRequest(BaseModel):
    """Top-level request for batch speech generation.
    Fields act as shared defaults; per-item overrides win."""

    model: str | None = None
    items: list[SpeechBatchItem] = Field(..., min_length=1, max_length=32)
    voice: str | None = None
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    max_new_tokens: int | None = None
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)
    repetition_penalty: float = 1.2
    exaggeration: float = 0.5
    max_tokens: int = 512


class SpeechBatchItemResult(BaseModel):
    index: int
    status: Literal["success", "error"]
    audio_data: str | None = None  # base64
    media_type: str | None = None
    error: str | None = None


class BatchSpeechResponse(BaseModel):
    id: str
    results: list[SpeechBatchItemResult]
    total: int
    succeeded: int
    failed: int


# ---------------------------------------------------------------------------
#  WebSocket streaming session config
# ---------------------------------------------------------------------------


class StreamingSpeechSessionConfig(BaseModel):
    """Configuration sent as the first WebSocket message for streaming TTS."""

    model: str | None = None
    voice: str | None = None
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)
    max_new_tokens: int | None = Field(default=None, ge=1)
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    speaker_embedding: list[float] | None = Field(
        default=None,
        max_length=_MAX_EMBEDDING_DIM,
    )
    stream_audio: bool = Field(
        default=False,
        description="If true, send raw PCM audio chunks progressively over WebSocket.",
    )
    split_granularity: Literal["sentence", "clause"] = Field(
        default="sentence",
        description="Text splitting: 'sentence' or 'clause' (adds CJK comma/semicolon)",
    )

    @model_validator(mode="after")
    def validate_streaming_constraints(self) -> "StreamingSpeechSessionConfig":
        if self.stream_audio:
            if self.response_format != "pcm":
                raise ValueError(
                    "WebSocket streaming (stream_audio=true) requires response_format='pcm'. "
                    f"Got response_format='{self.response_format}'."
                )
            if self.speed is None:
                self.speed = 1.0
            elif self.speed != 1.0:
                raise ValueError(
                    "Speed adjustment not supported with stream_audio=true."
                )
        return self


# ---------------------------------------------------------------------------
#  Audio response helpers
# ---------------------------------------------------------------------------


class CreateAudio(BaseModel):
    """Internal audio object for format conversion."""

    audio_tensor: np.ndarray
    sample_rate: int = 24000
    response_format: str = "wav"
    speed: float = 1.0
    stream_format: Literal["sse", "audio"] | None = "audio"
    base64_encode: bool = True

    class Config:
        arbitrary_types_allowed = True


class AudioResponse(BaseModel):
    audio_data: bytes | str
    media_type: str


# ---------------------------------------------------------------------------
#  Voice management (GET/POST/DELETE /v1/audio/voices)
# ---------------------------------------------------------------------------


class VoiceInfo(BaseModel):
    name: str
    description: str | None = None
    model_type: str | None = None
    sample_rate: int | None = None
    created_at: int | None = None
    file_size: int | None = None


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo]


class VoiceUploadResponse(BaseModel):
    name: str
    status: str = "uploaded"
    message: str | None = None


class VoiceDeleteResponse(BaseModel):
    name: str
    status: str = "deleted"
