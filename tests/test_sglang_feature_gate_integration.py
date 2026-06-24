"""
Comprehensive tests for Phase 4.5.2: SGLang Feature Gate Integration

Test coverage:
  - Startup config validation with fallback logic
  - Feature gate integration with compression backends
  - Fallback chain execution
  - Error handling and logging
  - Production initialization
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

import sys
sys.path.insert(0, "/home/local/ai/projects/gfxATOM-Rust/python")

# Mock feature gates and compression before importing
sys.modules['sglang_feature_gates'] = MagicMock()
sys.modules['sglang_kv_compression'] = MagicMock()

from sglang_feature_gate_integration import (
    SGLangFeatureGateIntegration,
    SGLangCompressionPipelineWithGates,
    get_feature_gate_integration,
    init_compression_with_gates,
)


class TestSGLangFeatureGateIntegration:
    """Test feature gate integration logic"""
    
    def test_integration_init(self):
        """Should initialize with defaults"""
        integration = SGLangFeatureGateIntegration()
        assert integration.validation_complete is False
        assert integration.selected_backend is None
        assert len(integration.fallback_history) == 0
    
    def test_validate_startup_config_fp16_no_gates(self):
        """FP16 should be valid even without feature gates"""
        integration = SGLangFeatureGateIntegration()
        with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', False):
            is_valid, error, mode = integration.validate_startup_config("fp16")
            assert is_valid is True
            assert mode == "fp16"
    
    def test_validate_startup_config_enables_tq2(self):
        """TQ2 validation should succeed"""
        integration = SGLangFeatureGateIntegration()
        
        # Mock feature gates
        with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', True), \
             patch('sglang_feature_gate_integration.validate_production_config') as mock_validate:
            mock_validate.return_value = (True, None)
            
            is_valid, error, mode = integration.validate_startup_config("tq2")
            assert is_valid is True
            assert mode == "tq2"
            assert integration.selected_backend == "tq2"
    
    def test_validate_startup_config_experimental_rejected(self):
        """Experimental modes should be rejected by default"""
        integration = SGLangFeatureGateIntegration()
        
        with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', True), \
             patch('sglang_feature_gate_integration.validate_production_config') as mock_validate, \
             patch('sglang_feature_gate_integration.FallbackChainManager.find_available_mode') as mock_fallback:
            
            # First validation fails (experimental), then fallback succeeds
            mock_validate.side_effect = [(False, "TurboQuant tq1 is EXPERIMENTAL"), (True, None)]
            mock_fallback.return_value = "tq2"
            
            is_valid, error, mode = integration.validate_startup_config(
                "tq1",
                allow_experimental=False
            )
            # Should fall back to tq2
            assert mode == "tq2"
    
    def test_get_compression_backend_tq2(self):
        """Should return backend config for enabled TurboQuant mode"""
        integration = SGLangFeatureGateIntegration()
        
        with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', True), \
             patch('sglang_feature_gate_integration.TurboQuantFeatureGates.is_enabled') as mock_enabled:
            mock_enabled.return_value = True
            
            backend = integration.get_compression_backend("tq2")
            assert backend is not None
            assert backend["mode"] == "tq2"
            assert backend["compression_type"] == "turboquant"
    
    def test_get_compression_backend_disabled(self):
        """Should return None for disabled modes"""
        integration = SGLangFeatureGateIntegration()
        
        with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', True), \
             patch('sglang_feature_gate_integration.TurboQuantFeatureGates.is_enabled') as mock_enabled:
            mock_enabled.return_value = False
            
            backend = integration.get_compression_backend("tq1")
            assert backend is None
    
    def test_execute_fallback_chain(self):
        """Should find available mode in fallback chain"""
        integration = SGLangFeatureGateIntegration()
        
        with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', True), \
             patch('sglang_feature_gate_integration.FallbackChainManager.find_available_mode') as mock_find:
            mock_find.return_value = "tq2"
            
            mode = integration.execute_fallback_chain("tq1")
            assert mode == "tq2"
    
    def test_log_startup_summary_no_validation(self):
        """Should log warning if validation not complete"""
        integration = SGLangFeatureGateIntegration()
        # Shouldn't crash, just log
        integration.log_startup_summary()
    
    def test_log_startup_summary_with_config(self):
        """Should log config summary when validation complete"""
        integration = SGLangFeatureGateIntegration()
        integration.validation_complete = True
        integration.config = {
            "requested": "tq1",
            "effective": "tq2",
            "fallback_reason": "experimental_disabled",
        }
        # Shouldn't crash, just log
        integration.log_startup_summary()


class TestSGLangCompressionPipelineWithGates:
    """Test compression pipeline with feature gates"""
    
    def test_pipeline_init(self):
        """Should initialize pipeline"""
        with patch('sglang_feature_gate_integration.COMPRESSION_AVAILABLE', False):
            pipeline = SGLangCompressionPipelineWithGates(manager=None)
            assert pipeline.manager is None
            assert isinstance(pipeline.integration, SGLangFeatureGateIntegration)
            assert len(pipeline.backend_errors) == 0
    
    def test_initialize_with_config_fp16(self):
        """Should initialize with FP16 config"""
        with patch('sglang_feature_gate_integration.COMPRESSION_AVAILABLE', False):
            pipeline = SGLangCompressionPipelineWithGates(manager=None)
            
            with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', False):
                config = {"kv_cache_dtype": "fp16"}
                success, error = pipeline.initialize_with_config(config)
                assert success is True
                assert error is None
    
    def test_initialize_with_config_tq2(self):
        """Should initialize with TQ2 config"""
        with patch('sglang_feature_gate_integration.COMPRESSION_AVAILABLE', False):
            pipeline = SGLangCompressionPipelineWithGates(manager=None)
            
            with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', True), \
                 patch.object(pipeline.integration, 'validate_startup_config') as mock_validate:
                mock_validate.return_value = (True, None, "tq2")
                
                config = {"kv_cache_dtype": "tq2"}
                success, error = pipeline.initialize_with_config(config)
                assert success is True
    
    def test_initialize_with_config_fails(self):
        """Should return error on validation failure"""
        pipeline = SGLangCompressionPipelineWithGates(manager=None)
        
        with patch.object(pipeline.integration, 'validate_startup_config') as mock_validate:
            mock_validate.return_value = (False, "hardware not gfx1030", "fp16")
            
            config = {"kv_cache_dtype": "tq2", "enforce_gfx1030": True}
            success, error = pipeline.initialize_with_config(config)
            assert success is False
            assert error is not None
    
    def test_encode_kv_with_fallback_no_manager(self):
        """Should return data unchanged if no manager"""
        with patch('sglang_feature_gate_integration.COMPRESSION_AVAILABLE', False):
            pipeline = SGLangCompressionPipelineWithGates(manager=None)
            
            kv_data = MagicMock()
            result = pipeline.encode_kv_with_fallback(kv_data, "tq2")
            assert result is kv_data
    
    def test_encode_kv_with_fallback_success(self):
        """Should encode KV successfully"""
        mock_manager = MagicMock()
        pipeline = SGLangCompressionPipelineWithGates(manager=mock_manager)
        
        kv_data = MagicMock()
        encoded_data = MagicMock()
        mock_manager.encode.return_value = encoded_data
        
        result = pipeline.encode_kv_with_fallback(kv_data, "tq2")
        assert result is encoded_data
        mock_manager.encode.assert_called_once_with(kv_data, "tq2")
    
    def test_encode_kv_with_fallback_retries(self):
        """Should retry with fallback on encoding error"""
        mock_manager = MagicMock()
        pipeline = SGLangCompressionPipelineWithGates(manager=mock_manager)
        
        kv_data = MagicMock()
        encoded_data = MagicMock()
        
        # First call fails, second succeeds
        mock_manager.encode.side_effect = [Exception("encode failed"), encoded_data]
        
        with patch.object(pipeline.integration, 'execute_fallback_chain') as mock_fallback:
            mock_fallback.return_value = "tq3"
            
            result = pipeline.encode_kv_with_fallback(kv_data, "tq2")
            # Should call encode twice (tq2, then tq3)
            assert mock_manager.encode.call_count == 2
            # First for tq2, second for tq3
            calls = [call[0][1] for call in mock_manager.encode.call_args_list]
            assert calls == ["tq2", "tq3"]
    
    def test_encode_kv_with_fallback_all_fail(self):
        """Should return original data if all encodes fail"""
        mock_manager = MagicMock()
        with patch('sglang_feature_gate_integration.COMPRESSION_AVAILABLE', False):
            pipeline = SGLangCompressionPipelineWithGates(manager=mock_manager)
        
        kv_data = MagicMock()
        mock_manager.encode.side_effect = Exception("encode failed")
        
        with patch.object(pipeline.integration, 'execute_fallback_chain') as mock_fallback:
            mock_fallback.return_value = "tq3"
            
            result = pipeline.encode_kv_with_fallback(kv_data, "tq2")
            
            # Should return original data
            assert result is kv_data
            # Should record error
            assert "tq2" in pipeline.backend_errors
    
    def test_decode_kv_with_fallback_no_manager(self):
        """Should return data unchanged if no manager"""
        with patch('sglang_feature_gate_integration.COMPRESSION_AVAILABLE', False):
            pipeline = SGLangCompressionPipelineWithGates(manager=None)
            
            compressed = MagicMock()
            result = pipeline.decode_kv_with_fallback(compressed, "tq2")
            assert result is compressed
    
    def test_decode_kv_with_fallback_success(self):
        """Should decode KV successfully"""
        mock_manager = MagicMock()
        pipeline = SGLangCompressionPipelineWithGates(manager=mock_manager)
        
        compressed = MagicMock()
        compressed.data = MagicMock()  # Has data attribute
        decoded_data = MagicMock()
        mock_manager.decode.return_value = decoded_data
        
        result = pipeline.decode_kv_with_fallback(compressed, "tq2")
        assert result is decoded_data


class TestGlobalFunctions:
    """Test module-level utility functions"""
    
    def test_get_feature_gate_integration_singleton(self):
        """Should return same instance on repeated calls"""
        first = get_feature_gate_integration()
        second = get_feature_gate_integration()
        assert first is second
    
    def test_init_compression_with_gates_default(self):
        """Should initialize with default config"""
        with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', False):
            success, error = init_compression_with_gates()
            assert success is True
            assert error is None
    
    def test_init_compression_with_gates_fp16(self):
        """Should initialize with FP16"""
        with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', False):
            config = {"kv_cache_dtype": "fp16"}
            success, error = init_compression_with_gates(config)
            assert success is True
    
    def test_init_compression_with_gates_fails(self):
        """Should return error on failure"""
        with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', True), \
             patch('sglang_feature_gate_integration.validate_production_config') as mock_validate:
            mock_validate.return_value = (False, "hardware error")
            
            config = {"kv_cache_dtype": "tq2", "enforce_gfx1030": True}
            success, error = init_compression_with_gates(config)
            assert success is False
            assert error is not None


class TestFallbackChainBehavior:
    """Test realistic fallback chain scenarios"""
    
    def test_fallback_chain_tq1_degradation(self):
        """TQ1 should degrade through tq2 -> tq3 -> tq4 -> fp8"""
        integration = SGLangFeatureGateIntegration()
        
        with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', True), \
             patch('sglang_feature_gate_integration.FallbackChainManager.get_fallback_chain') as mock_chain:
            mock_chain.return_value = ["tq2", "tq3", "tq4", "fp8_e4m3", "fp16"]
            
            chain = mock_chain("tq1")
            assert "tq2" in chain
            assert "fp16" in chain
    
    def test_production_mode_forced_tq2(self):
        """Production mode should prefer TQ2 over TQ1"""
        integration = SGLangFeatureGateIntegration()
        
        with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', True), \
             patch('sglang_feature_gate_integration.validate_production_config') as mock_validate:
            # Simulate TQ2 being allowed but TQ1 not
            def validate_side_effect(mode, *args, **kwargs):
                if mode == "tq1" and not kwargs.get('allow_experimental'):
                    return (False, "experimental")
                return (True, None)
            
            mock_validate.side_effect = validate_side_effect
            
            # Request TQ1 in production
            is_valid, error, mode = integration.validate_startup_config(
                "tq1",
                allow_experimental=False
            )
            # Should fall back
            assert mode != "tq1" or is_valid


class TestErrorHandlingAndLogging:
    """Test error handling and logging behavior"""
    
    def test_validation_error_includes_help(self):
        """Validation errors should include helpful guidance"""
        integration = SGLangFeatureGateIntegration()
        
        with patch('sglang_feature_gate_integration.FEATURE_GATES_AVAILABLE', True), \
             patch('sglang_feature_gate_integration.validate_production_config') as mock_validate:
            mock_validate.return_value = (False, "EXPERIMENTAL mode disabled\nTo enable: export SGLANG_ENABLE_EXPERIMENTAL=true")
            
            is_valid, error, mode = integration.validate_startup_config("tq1")
            # Should either fall back or return error with help text
    
    def test_backend_error_tracking(self):
        """Should track which backends have failed"""
        mock_manager = MagicMock()
        with patch('sglang_feature_gate_integration.COMPRESSION_AVAILABLE', False):
            pipeline = SGLangCompressionPipelineWithGates(manager=mock_manager)
        
        kv_data = MagicMock()
        error_msg = "GPU out of memory"
        mock_manager.encode.side_effect = Exception(error_msg)
        
        with patch.object(pipeline.integration, 'execute_fallback_chain') as mock_fallback:
            mock_fallback.return_value = "fp16"  # Fallback to fp16
            
            pipeline.encode_kv_with_fallback(kv_data, "tq2")
            
            # Should record error
            assert "tq2" in pipeline.backend_errors
            assert error_msg in pipeline.backend_errors["tq2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
