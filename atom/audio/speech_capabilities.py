# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Speech model capability validators for ATOM multi-model TTS/ASR serving.

Provides explicit model/task capability checks to ensure unsupported combinations
fail fast with clear error messages rather than silently degrading.

Supports task types:
- speech_synthesis (TTS): Text-to-speech generation
- speech_recognition (ASR): Audio-to-text transcription
- voice_cloning (VC): Voice conversion with speaker reference
- streaming_synthesis: Incremental/streaming TTS output
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import HTTPException

logger = logging.getLogger("atom.audio.capabilities")


class SpeechTask(str, Enum):
    """Supported speech processing tasks."""
    SYNTHESIS = "speech_synthesis"
    RECOGNITION = "speech_recognition"
    CLONING = "voice_cloning"
    STREAMING = "streaming_synthesis"


class ModelBackend(str, Enum):
    """Available model backends."""
    CHATTERBOX = "chatterbox"
    CHATTERBOX_TURBO = "chatterbox_turbo"
    MOONSHINE = "moonshine"
    QWEN_ASR = "qwen_asr"
    FISH_SPEECH = "fish_speech"
    VOCPM2 = "vocpm2"
    ATOM_VLLM = "atom_vllm"
    ONNX = "onnx"
    AUTO = "auto"


@dataclass(frozen=True)
class ModelCapabilities:
    """Declared capabilities and constraints for a speech model."""
    
    name: str
    backend: ModelBackend
    tasks: set[SpeechTask]
    max_text_length: int | None = None
    max_audio_duration_sec: float | None = None
    supports_streaming: bool = False
    supports_ref_audio: bool = False
    supported_voices: set[str] | None = None
    supported_sample_rates: set[int] | None = None
    requires_backend_available: bool = True


class SpeechCapabilityRegistry:
    """Central registry of model capabilities and constraints.
    
    Used to validate that requested model/task combinations are supported
    before attempting execution.
    """

    # Registry of known models and their capabilities
    _KNOWN_MODELS: dict[str, ModelCapabilities] = {
        # TTS Models
        "chatterbox": ModelCapabilities(
            name="chatterbox",
            backend=ModelBackend.CHATTERBOX,
            tasks={SpeechTask.SYNTHESIS, SpeechTask.STREAMING},
            max_text_length=1000,
            max_audio_duration_sec=120.0,
            supports_streaming=True,
            supports_ref_audio=True,
            supported_voices={"default", "af_bella"},
            supported_sample_rates={24000, 22050},
        ),
        "chatterbox-turbo": ModelCapabilities(
            name="chatterbox-turbo",
            backend=ModelBackend.CHATTERBOX_TURBO,
            tasks={SpeechTask.SYNTHESIS, SpeechTask.STREAMING},
            max_text_length=1000,
            max_audio_duration_sec=120.0,
            supports_streaming=True,
            supports_ref_audio=True,
            supported_voices={"default", "af_bella"},
            supported_sample_rates={24000, 22050},
        ),
        # ASR Models
        "moonshine": ModelCapabilities(
            name="moonshine",
            backend=ModelBackend.MOONSHINE,
            tasks={SpeechTask.RECOGNITION},
            max_audio_duration_sec=300.0,
            supports_ref_audio=False,
            supported_sample_rates={16000},
        ),
        "qwen-asr": ModelCapabilities(
            name="qwen-asr",
            backend=ModelBackend.QWEN_ASR,
            tasks={SpeechTask.RECOGNITION},
            max_audio_duration_sec=300.0,
            supports_ref_audio=False,
            supported_sample_rates={16000},
        ),
        # Experimental models (disabled by default)
        "fish-speech": ModelCapabilities(
            name="fish-speech",
            backend=ModelBackend.FISH_SPEECH,
            tasks={SpeechTask.SYNTHESIS},
            max_text_length=500,
            max_audio_duration_sec=60.0,
            supports_streaming=False,
            supports_ref_audio=True,
            requires_backend_available=True,
        ),
        "vocpm2": ModelCapabilities(
            name="vocpm2",
            backend=ModelBackend.VOCPM2,
            tasks={SpeechTask.SYNTHESIS, SpeechTask.CLONING},
            max_text_length=1000,
            max_audio_duration_sec=120.0,
            supports_ref_audio=True,
            requires_backend_available=True,
        ),
    }

    def __init__(self):
        """Initialize capability registry."""
        self._custom_capabilities: dict[str, ModelCapabilities] = {}

    def register_model(self, capabilities: ModelCapabilities) -> None:
        """Register custom model capabilities.
        
        Args:
            capabilities: ModelCapabilities object defining model's capabilities.
        """
        name_lower = capabilities.name.lower()
        self._custom_capabilities[name_lower] = capabilities
        logger.debug(f"Registered custom model: {capabilities.name} "
                    f"tasks={capabilities.tasks}")

    def get_capabilities(self, model_name: str) -> ModelCapabilities | None:
        """Get capabilities for a model by name.
        
        Args:
            model_name: Model name (case-insensitive).
            
        Returns:
            ModelCapabilities if found, None otherwise.
        """
        name_lower = model_name.lower()
        
        # Check custom registrations first
        if name_lower in self._custom_capabilities:
            return self._custom_capabilities[name_lower]
        
        # Check known models
        if name_lower in self._KNOWN_MODELS:
            return self._KNOWN_MODELS[name_lower]
        
        # Fuzzy match
        for known_name, caps in self._KNOWN_MODELS.items():
            if name_lower in known_name.lower() or known_name.lower() in name_lower:
                return caps
        
        return None

    def list_models(self) -> list[str]:
        """List all available models."""
        models = list(self._KNOWN_MODELS.keys()) + list(self._custom_capabilities.keys())
        return sorted(set(models))

    def list_models_for_task(self, task: SpeechTask) -> list[str]:
        """List models that support a given task."""
        models = []
        
        for caps in self._KNOWN_MODELS.values():
            if task in caps.tasks:
                models.append(caps.name)
        
        for caps in self._custom_capabilities.values():
            if task in caps.tasks:
                models.append(caps.name)
        
        return sorted(set(models))


class SpeechCapabilityValidator:
    """Validates speech model requests against declared capabilities."""

    def __init__(self, registry: SpeechCapabilityRegistry | None = None):
        """Initialize validator with a capability registry.
        
        Args:
            registry: Optional custom registry. Uses global default if None.
        """
        self.registry = registry or _GLOBAL_REGISTRY

    def validate_model_available(
        self,
        model_name: str,
    ) -> ModelCapabilities:
        """Validate that a model is registered and available.
        
        Args:
            model_name: Model name to validate.
            
        Returns:
            ModelCapabilities for the model.
            
        Raises:
            ValueError: If model is unknown or unavailable.
        """
        capabilities = self.registry.get_capabilities(model_name)
        
        if capabilities is None:
            available = self.registry.list_models()
            raise ValueError(
                f"Model '{model_name}' is not available. "
                f"Available models: {available}"
            )
        
        if capabilities.requires_backend_available:
            # TODO: Add backend availability checks (sglang, vllm, etc.)
            pass
        
        return capabilities

    def validate_task_supported(
        self,
        model_name: str,
        task: SpeechTask | str,
    ) -> ModelCapabilities:
        """Validate that a model supports a specific task.
        
        Args:
            model_name: Model name.
            task: Task to validate (SpeechTask enum or string).
            
        Returns:
            ModelCapabilities for the model.
            
        Raises:
            ValueError: If model doesn't support the task.
        """
        # Normalize task
        if isinstance(task, str):
            try:
                task = SpeechTask(task)
            except ValueError:
                raise ValueError(
                    f"Unknown task '{task}'. "
                    f"Supported tasks: {[t.value for t in SpeechTask]}"
                )
        
        capabilities = self.validate_model_available(model_name)
        
        if task not in capabilities.tasks:
            models_for_task = self.registry.list_models_for_task(task)
            raise ValueError(
                f"Model '{model_name}' does not support task '{task.value}'. "
                f"Models supporting {task.value}: {models_for_task}"
            )
        
        return capabilities

    def validate_text_length(
        self,
        model_name: str,
        text: str,
    ) -> None:
        """Validate that input text is within model limits.
        
        Args:
            model_name: Model name.
            text: Input text to validate.
            
        Raises:
            ValueError: If text exceeds model's max_text_length.
        """
        capabilities = self.validate_model_available(model_name)
        
        if capabilities.max_text_length is not None:
            if len(text) > capabilities.max_text_length:
                raise ValueError(
                    f"Text length {len(text)} exceeds model limit "
                    f"{capabilities.max_text_length} for '{model_name}'"
                )

    def validate_ref_audio_supported(
        self,
        model_name: str,
    ) -> None:
        """Validate that a model supports reference audio (for voice cloning).
        
        Args:
            model_name: Model name.
            
        Raises:
            ValueError: If model doesn't support reference audio.
        """
        capabilities = self.validate_model_available(model_name)
        
        if not capabilities.supports_ref_audio:
            ref_audio_models = [
                caps.name for caps in self.registry._KNOWN_MODELS.values()
                if caps.supports_ref_audio
            ]
            raise ValueError(
                f"Model '{model_name}' does not support reference audio/voice cloning. "
                f"Models supporting voice cloning: {ref_audio_models}"
            )

    def validate_streaming_supported(
        self,
        model_name: str,
    ) -> None:
        """Validate that a model supports streaming output.
        
        Args:
            model_name: Model name.
            
        Raises:
            ValueError: If model doesn't support streaming.
        """
        capabilities = self.validate_model_available(model_name)
        
        if not capabilities.supports_streaming:
            streaming_models = [
                caps.name for caps in self.registry._KNOWN_MODELS.values()
                if caps.supports_streaming
            ]
            raise ValueError(
                f"Model '{model_name}' does not support streaming output. "
                f"Models supporting streaming: {streaming_models}"
            )


# Global default registry and validator
_GLOBAL_REGISTRY = SpeechCapabilityRegistry()
_GLOBAL_VALIDATOR = SpeechCapabilityValidator(_GLOBAL_REGISTRY)


def get_global_registry() -> SpeechCapabilityRegistry:
    """Get the global capability registry."""
    return _GLOBAL_REGISTRY


def get_global_validator() -> SpeechCapabilityValidator:
    """Get the global capability validator."""
    return _GLOBAL_VALIDATOR


def register_custom_model(capabilities: ModelCapabilities) -> None:
    """Register a custom model with the global registry.
    
    Args:
        capabilities: ModelCapabilities object defining model's capabilities.
    """
    _GLOBAL_REGISTRY.register_model(capabilities)


# Exported convenience functions
def validate_model_task(model_name: str, task: SpeechTask | str) -> ModelCapabilities:
    """Validate a model/task combination using the global validator.
    
    Args:
        model_name: Model name.
        task: Task name or SpeechTask enum.
        
    Returns:
        ModelCapabilities if valid.
        
    Raises:
        ValueError: If combination is unsupported.
    """
    return _GLOBAL_VALIDATOR.validate_task_supported(model_name, task)


def validate_ref_audio(model_name: str) -> None:
    """Validate that a model supports reference audio.
    
    Args:
        model_name: Model name.
        
    Raises:
        ValueError: If model doesn't support reference audio.
    """
    _GLOBAL_VALIDATOR.validate_ref_audio_supported(model_name)


def validate_streaming(model_name: str) -> None:
    """Validate that a model supports streaming.
    
    Args:
        model_name: Model name.
        
    Raises:
        ValueError: If model doesn't support streaming.
    """
    _GLOBAL_VALIDATOR.validate_streaming_supported(model_name)


def list_available_models() -> list[str]:
    """List all available speech models."""
    return _GLOBAL_REGISTRY.list_models()


def list_models_for_task(task: SpeechTask | str) -> list[str]:
    """List models that support a given task.
    
    Args:
        task: Task name or SpeechTask enum.
        
    Returns:
        List of model names supporting the task.
    """
    if isinstance(task, str):
        task = SpeechTask(task)
    return _GLOBAL_REGISTRY.list_models_for_task(task)
