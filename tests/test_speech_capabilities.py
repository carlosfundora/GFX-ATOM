# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for speech model capability validators.

Validates:
- Model registration and discovery
- Task support validation
- Reference audio support checks
- Streaming support checks
- Text length constraints
- Error messages for unsupported combinations
"""

import pytest

from atom.audio.speech_capabilities import (
    ModelBackend,
    ModelCapabilities,
    SpeechCapabilityRegistry,
    SpeechCapabilityValidator,
    SpeechTask,
    get_global_registry,
    get_global_validator,
    list_available_models,
    list_models_for_task,
    register_custom_model,
    validate_model_task,
    validate_ref_audio,
    validate_streaming,
)


class TestSpeechTaskEnum:
    """Test SpeechTask enumeration."""

    def test_task_values_defined(self):
        """Verify all required speech tasks are defined."""
        assert SpeechTask.SYNTHESIS.value == "speech_synthesis"
        assert SpeechTask.RECOGNITION.value == "speech_recognition"
        assert SpeechTask.CLONING.value == "voice_cloning"
        assert SpeechTask.STREAMING.value == "streaming_synthesis"

    def test_task_enum_iteration(self):
        """Verify all tasks are iterable."""
        tasks = list(SpeechTask)
        assert len(tasks) == 4


class TestModelCapabilities:
    """Test ModelCapabilities dataclass."""

    def test_create_basic_capabilities(self):
        """Create and validate basic model capabilities."""
        caps = ModelCapabilities(
            name="test-model",
            backend=ModelBackend.CHATTERBOX,
            tasks={SpeechTask.SYNTHESIS},
        )
        
        assert caps.name == "test-model"
        assert caps.backend == ModelBackend.CHATTERBOX
        assert SpeechTask.SYNTHESIS in caps.tasks
        assert caps.requires_backend_available is True

    def test_capabilities_with_constraints(self):
        """Create capabilities with text/audio constraints."""
        caps = ModelCapabilities(
            name="constrained-model",
            backend=ModelBackend.MOONSHINE,
            tasks={SpeechTask.RECOGNITION},
            max_text_length=500,
            max_audio_duration_sec=60.0,
            supported_sample_rates={16000, 22050},
        )
        
        assert caps.max_text_length == 500
        assert caps.max_audio_duration_sec == 60.0
        assert 16000 in caps.supported_sample_rates


class TestSpeechCapabilityRegistry:
    """Test capability registry and model discovery."""

    def test_registry_has_known_models(self):
        """Verify registry has known models on init."""
        registry = SpeechCapabilityRegistry()
        models = registry.list_models()
        
        assert "chatterbox" in models
        assert "moonshine" in models
        assert len(models) > 0

    def test_get_capabilities_exact_match(self):
        """Retrieve capabilities with exact model name."""
        registry = SpeechCapabilityRegistry()
        caps = registry.get_capabilities("chatterbox")
        
        assert caps is not None
        assert caps.name == "chatterbox"
        assert SpeechTask.SYNTHESIS in caps.tasks

    def test_get_capabilities_case_insensitive(self):
        """Retrieve capabilities with case-insensitive lookup."""
        registry = SpeechCapabilityRegistry()
        caps = registry.get_capabilities("CHATTERBOX")
        
        assert caps is not None
        assert caps.name == "chatterbox"

    def test_get_capabilities_unknown_returns_none(self):
        """Unknown models return None."""
        registry = SpeechCapabilityRegistry()
        caps = registry.get_capabilities("unknown-model-xyz")
        
        assert caps is None

    def test_register_custom_model(self):
        """Register a custom model."""
        registry = SpeechCapabilityRegistry()
        custom_caps = ModelCapabilities(
            name="custom-tts",
            backend=ModelBackend.AUTO,
            tasks={SpeechTask.SYNTHESIS},
        )
        
        registry.register_model(custom_caps)
        
        retrieved = registry.get_capabilities("custom-tts")
        assert retrieved is not None
        assert retrieved.name == "custom-tts"

    def test_list_models_for_task(self):
        """List models supporting a specific task."""
        registry = SpeechCapabilityRegistry()
        synthesis_models = registry.list_models_for_task(SpeechTask.SYNTHESIS)
        
        assert "chatterbox" in synthesis_models
        assert "moonshine" not in synthesis_models
        
        recognition_models = registry.list_models_for_task(SpeechTask.RECOGNITION)
        assert "moonshine" in recognition_models
        assert "chatterbox" not in recognition_models  # chatterbox only does synthesis


class TestSpeechCapabilityValidator:
    """Test capability validation logic."""

    @pytest.fixture
    def validator(self):
        """Create a validator with a fresh registry."""
        registry = SpeechCapabilityRegistry()
        return SpeechCapabilityValidator(registry)

    def test_validate_model_available_success(self, validator):
        """Validate model availability."""
        caps = validator.validate_model_available("chatterbox")
        assert caps.name == "chatterbox"

    def test_validate_model_available_unknown_raises(self, validator):
        """Unknown model raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            validator.validate_model_available("unknown-xyz")

    def test_validate_task_supported_success(self, validator):
        """Validate task is supported."""
        caps = validator.validate_task_supported("chatterbox", SpeechTask.SYNTHESIS)
        assert SpeechTask.SYNTHESIS in caps.tasks

    def test_validate_task_supported_unsupported_raises(self, validator):
        """Unsupported task raises ValueError."""
        with pytest.raises(ValueError, match="does not support task"):
            validator.validate_task_supported("moonshine", SpeechTask.SYNTHESIS)

    def test_validate_task_supported_string_task(self, validator):
        """Accept task as string."""
        caps = validator.validate_task_supported("chatterbox", "speech_synthesis")
        assert caps is not None

    def test_validate_task_supported_invalid_task_string_raises(self, validator):
        """Invalid task string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            validator.validate_task_supported("chatterbox", "invalid_task")

    def test_validate_text_length_within_limit(self, validator):
        """Text within model limit succeeds."""
        text = "Hello world"
        # Should not raise
        validator.validate_text_length("chatterbox", text)

    def test_validate_text_length_exceeds_limit_raises(self, validator):
        """Text exceeding model limit raises ValueError."""
        text = "x" * 2000  # Exceeds chatterbox limit
        with pytest.raises(ValueError, match="exceeds model limit"):
            validator.validate_text_length("chatterbox", text)

    def test_validate_ref_audio_supported(self, validator):
        """Model with ref audio support validates."""
        # Should not raise
        validator.validate_ref_audio_supported("chatterbox")

    def test_validate_ref_audio_unsupported_raises(self, validator):
        """Model without ref audio support raises ValueError."""
        with pytest.raises(ValueError, match="does not support reference audio"):
            validator.validate_ref_audio_supported("moonshine")

    def test_validate_streaming_supported(self, validator):
        """Model with streaming support validates."""
        # Should not raise
        validator.validate_streaming_supported("chatterbox")

    def test_validate_streaming_unsupported_raises(self, validator):
        """Model without streaming support raises ValueError."""
        with pytest.raises(ValueError, match="does not support streaming output"):
            validator.validate_streaming_supported("moonshine")


class TestGlobalRegistry:
    """Test global registry and convenience functions."""

    def test_get_global_registry(self):
        """Get global registry."""
        registry = get_global_registry()
        assert registry is not None
        assert len(registry.list_models()) > 0

    def test_get_global_validator(self):
        """Get global validator."""
        validator = get_global_validator()
        assert validator is not None

    def test_list_available_models(self):
        """List all available models."""
        models = list_available_models()
        assert "chatterbox" in models
        assert isinstance(models, list)

    def test_list_models_for_task_synthesis(self):
        """List models supporting synthesis."""
        models = list_models_for_task(SpeechTask.SYNTHESIS)
        assert "chatterbox" in models

    def test_list_models_for_task_recognition(self):
        """List models supporting recognition."""
        models = list_models_for_task(SpeechTask.RECOGNITION)
        assert "moonshine" in models

    def test_validate_model_task_convenience(self):
        """Use convenience validation function."""
        caps = validate_model_task("chatterbox", SpeechTask.SYNTHESIS)
        assert caps.name == "chatterbox"

    def test_validate_model_task_unsupported_raises(self):
        """Unsupported combination raises via convenience function."""
        with pytest.raises(ValueError):
            validate_model_task("moonshine", SpeechTask.SYNTHESIS)

    def test_validate_ref_audio_convenience(self):
        """Use ref audio validation convenience function."""
        # Should not raise
        validate_ref_audio("chatterbox")

    def test_validate_ref_audio_unsupported_raises(self):
        """Unsupported ref audio raises via convenience function."""
        with pytest.raises(ValueError):
            validate_ref_audio("moonshine")

    def test_validate_streaming_convenience(self):
        """Use streaming validation convenience function."""
        # Should not raise
        validate_streaming("chatterbox")

    def test_validate_streaming_unsupported_raises(self):
        """Unsupported streaming raises via convenience function."""
        with pytest.raises(ValueError):
            validate_streaming("moonshine")

    def test_register_custom_model_global(self):
        """Register custom model in global registry."""
        custom_caps = ModelCapabilities(
            name="test-custom-model",
            backend=ModelBackend.AUTO,
            tasks={SpeechTask.SYNTHESIS},
        )
        
        register_custom_model(custom_caps)
        
        # Verify it's in the global registry
        models = list_available_models()
        assert "test-custom-model" in models


class TestErrorMessages:
    """Test quality of error messages."""

    @pytest.fixture
    def validator(self):
        """Create a validator."""
        registry = SpeechCapabilityRegistry()
        return SpeechCapabilityValidator(registry)

    def test_unknown_model_error_lists_available(self, validator):
        """Error for unknown model lists available options."""
        try:
            validator.validate_model_available("unknown-xyz")
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "Available models" in error_msg
            assert "chatterbox" in error_msg

    def test_unsupported_task_error_lists_alternatives(self, validator):
        """Error for unsupported task lists models that support it."""
        try:
            validator.validate_task_supported("moonshine", SpeechTask.SYNTHESIS)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "does not support task" in error_msg
            assert "Models supporting" in error_msg

    def test_ref_audio_error_lists_alternatives(self, validator):
        """Error for unsupported ref audio lists models that support it."""
        try:
            validator.validate_ref_audio_supported("moonshine")
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "does not support reference audio" in error_msg
            assert "voice cloning" in error_msg

    def test_streaming_error_lists_alternatives(self, validator):
        """Error for unsupported streaming lists models that support it."""
        try:
            validator.validate_streaming_supported("moonshine")
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "does not support streaming output" in error_msg
            assert "Models supporting streaming" in error_msg
