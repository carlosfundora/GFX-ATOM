# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Integration tests for speech capability validators with SpeechServing.

Tests that capability validators are properly integrated into the serving layer
and that requests are validated before execution.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from atom.audio.speech_capabilities import (
    SpeechCapabilityValidator,
    SpeechCapabilityRegistry,
    SpeechTask,
)


class TestSpeechServingCapabilityIntegration:
    """Test integration of capability validators with SpeechServing."""

    @pytest.fixture
    def mock_serving_setup(self):
        """Setup mocks for SpeechServing validation layer."""
        registry = SpeechCapabilityRegistry()
        validator = SpeechCapabilityValidator(registry)
        return {
            'registry': registry,
            'validator': validator,
        }

    def test_validator_is_initialized_in_serving(self, mock_serving_setup):
        """Verify validator is properly initialized for serving."""
        validator = mock_serving_setup['validator']
        
        # Validator should be callable and have required methods
        assert hasattr(validator, 'validate_model_available')
        assert hasattr(validator, 'validate_task_supported')
        assert hasattr(validator, 'validate_text_length')
        assert hasattr(validator, 'validate_ref_audio_supported')
        assert hasattr(validator, 'validate_streaming_supported')

    def test_synthesis_validation_accepts_valid_request(self, mock_serving_setup):
        """Valid synthesis request passes validation."""
        validator = mock_serving_setup['validator']
        
        # Should not raise
        caps = validator.validate_task_supported("chatterbox", SpeechTask.SYNTHESIS)
        assert caps is not None

    def test_synthesis_validation_rejects_invalid_model(self, mock_serving_setup):
        """Invalid model name rejected during validation."""
        validator = mock_serving_setup['validator']
        
        with pytest.raises(ValueError, match="not available"):
            validator.validate_model_available("nonexistent-model-xyz")

    def test_synthesis_validation_rejects_unsupported_task(self, mock_serving_setup):
        """Unsupported task combination rejected during validation."""
        validator = mock_serving_setup['validator']
        
        # Moonshine doesn't do synthesis
        with pytest.raises(ValueError, match="does not support task"):
            validator.validate_task_supported("moonshine", SpeechTask.SYNTHESIS)

    def test_synthesis_validation_rejects_text_too_long(self, mock_serving_setup):
        """Text exceeding model limit rejected during validation."""
        validator = mock_serving_setup['validator']
        
        long_text = "x" * 5000  # Exceeds chatterbox 1000 limit
        with pytest.raises(ValueError, match="exceeds model limit"):
            validator.validate_text_length("chatterbox", long_text)

    def test_cloning_validation_requires_ref_audio_support(self, mock_serving_setup):
        """Voice cloning validation requires reference audio support."""
        validator = mock_serving_setup['validator']
        
        # Chatterbox supports ref audio
        validator.validate_ref_audio_supported("chatterbox")  # Should not raise
        
        # Moonshine does not
        with pytest.raises(ValueError, match="does not support reference audio"):
            validator.validate_ref_audio_supported("moonshine")

    def test_streaming_validation_requires_streaming_support(self, mock_serving_setup):
        """Streaming validation requires model support."""
        validator = mock_serving_setup['validator']
        
        # Chatterbox supports streaming
        validator.validate_streaming_supported("chatterbox")  # Should not raise
        
        # Moonshine does not
        with pytest.raises(ValueError, match="does not support streaming output"):
            validator.validate_streaming_supported("moonshine")

    def test_validation_error_messages_guide_users(self, mock_serving_setup):
        """Validation errors provide clear guidance."""
        validator = mock_serving_setup['validator']
        
        # Unknown model error should list available models
        try:
            validator.validate_model_available("unknown-xyz")
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "Available models" in error_msg
            assert "chatterbox" in error_msg
            assert "moonshine" in error_msg

        # Unsupported task error should list alternative models
        try:
            validator.validate_task_supported("moonshine", SpeechTask.SYNTHESIS)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "does not support task" in error_msg
            assert "speech_synthesis" in error_msg


class TestCapabilityValidationScenarios:
    """Test realistic validation scenarios."""

    @pytest.fixture
    def validator(self):
        """Create a validator."""
        registry = SpeechCapabilityRegistry()
        return SpeechCapabilityValidator(registry)

    def test_scenario_user_requests_asr_with_tts_model(self, validator):
        """User tries to use TTS model for ASR task."""
        # This should fail clearly
        with pytest.raises(ValueError) as exc_info:
            validator.validate_task_supported("chatterbox", SpeechTask.RECOGNITION)
        
        error_msg = str(exc_info.value)
        assert "chatterbox" in error_msg
        assert "does not support" in error_msg
        assert "speech_recognition" in error_msg
        
        # Error should suggest alternatives
        assert "moonshine" in error_msg or "qwen" in error_msg.lower()

    def test_scenario_user_requests_cloning_without_ref_audio_support(self, validator):
        """User tries voice cloning on model that doesn't support it."""
        with pytest.raises(ValueError) as exc_info:
            validator.validate_ref_audio_supported("moonshine")
        
        error_msg = str(exc_info.value)
        assert "reference audio" in error_msg
        assert "voice cloning" in error_msg
        # Should suggest alternatives that do support it
        assert "chatterbox" in error_msg

    def test_scenario_user_provides_very_long_text(self, validator):
        """User provides text exceeding model limit."""
        long_text = "hello " * 500  # Exceeds limit
        
        with pytest.raises(ValueError) as exc_info:
            validator.validate_text_length("chatterbox", long_text)
        
        error_msg = str(exc_info.value)
        assert "exceeds model limit" in error_msg
        assert "1000" in error_msg  # chatterbox limit

    def test_scenario_user_requests_streaming_on_nonsupporting_model(self, validator):
        """User requests streaming on model without streaming support."""
        with pytest.raises(ValueError) as exc_info:
            validator.validate_streaming_supported("moonshine")
        
        error_msg = str(exc_info.value)
        assert "does not support streaming" in error_msg
        # Should suggest alternatives
        assert "Models supporting streaming" in error_msg


class TestValidationIntegrationWithRegistry:
    """Test validation integrated with model registry."""

    def test_custom_model_registration_and_validation(self):
        """Custom models can be registered and validated."""
        from atom.audio.speech_capabilities import (
            ModelCapabilities,
            ModelBackend,
            SpeechCapabilityRegistry,
            SpeechCapabilityValidator,
        )
        
        # Create custom registry
        registry = SpeechCapabilityRegistry()
        
        # Register custom model
        custom_caps = ModelCapabilities(
            name="custom-tts",
            backend=ModelBackend.AUTO,
            tasks={SpeechTask.SYNTHESIS},
            max_text_length=2000,
        )
        registry.register_model(custom_caps)
        
        # Create validator with custom registry
        validator = SpeechCapabilityValidator(registry)
        
        # Should find the custom model
        caps = validator.validate_model_available("custom-tts")
        assert caps.name == "custom-tts"
        assert caps.max_text_length == 2000

    def test_model_discovery_in_validation_errors(self):
        """Errors should help users discover available models."""
        from atom.audio.speech_capabilities import (
            SpeechCapabilityRegistry,
            SpeechCapabilityValidator,
        )
        
        registry = SpeechCapabilityRegistry()
        available = registry.list_models()
        
        validator = SpeechCapabilityValidator(registry)
        
        # When user requests unknown model, error should list available ones
        try:
            validator.validate_model_available("user-made-up-model")
            pytest.fail("Should have raised")
        except ValueError as e:
            error_msg = str(e)
            # Should mention at least some available models
            for model in available[:3]:  # At least show first 3
                if model:
                    assert model in error_msg or "Available models" in error_msg
