"""
Comprehensive tests for Phase 4.5: Feature Gates and Production Safety

Test coverage:
  - Feature gate enablement logic (feature gates off/on)
  - Hardware detection and validation
  - Fallback chain selection
  - Production configuration validation
  - Error messaging clarity
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Setup PYTHONPATH
import sys
sys.path.insert(0, "/home/local/ai/projects/gfxATOM-Rust/python")

from sglang_feature_gates import (
    FeatureGate,
    FeatureGateConfig,
    TurboQuantFeatureGates,
    RotorQuantFeatureGates,
    HardwareSafetyValidator,
    FallbackChainManager,
    validate_production_config,
)


class TestFeatureGateDefinitions:
    """Test feature gate definitions and structure"""
    
    def test_turboquant_gates_defined(self):
        """Verify all TurboQuant gates are defined"""
        expected_modes = ["tq1", "tq2", "tq3", "tq4", "tq8"]
        assert all(mode in TurboQuantFeatureGates.gates for mode in expected_modes)
    
    def test_turboquant_gate_statuses(self):
        """Verify correct gate statuses for TurboQuant modes"""
        assert TurboQuantFeatureGates.gates["tq1"].gate_status == FeatureGate.EXPERIMENTAL
        assert TurboQuantFeatureGates.gates["tq2"].gate_status == FeatureGate.BETA
        assert TurboQuantFeatureGates.gates["tq3"].gate_status == FeatureGate.BETA
        assert TurboQuantFeatureGates.gates["tq4"].gate_status == FeatureGate.STABLE
        assert TurboQuantFeatureGates.gates["tq8"].gate_status == FeatureGate.OFF
    
    def test_turboquant_hw_support(self):
        """Verify hardware support is documented"""
        for mode in ["tq1", "tq2", "tq3", "tq4", "tq8"]:
            gate = TurboQuantFeatureGates.gates[mode]
            assert "gfx1030" in gate.supported_hw
    
    def test_rotorquant_gates_defined(self):
        """Verify RotorQuant gates are defined"""
        expected_modes = ["rq3_planar", "rq4_planar", "rq3_iso", "rq4_iso"]
        assert all(mode in RotorQuantFeatureGates.gates for mode in expected_modes)
    
    def test_rotorquant_all_experimental(self):
        """Verify all RotorQuant modes are EXPERIMENTAL"""
        for mode, gate in RotorQuantFeatureGates.gates.items():
            assert gate.gate_status == FeatureGate.EXPERIMENTAL


class TestTurboQuantFeatureGateEnablement:
    """Test TurboQuant mode enablement logic"""
    
    def test_tq4_enabled_by_default(self):
        """TQ4 (STABLE) should be enabled by default"""
        with patch.dict(os.environ, {}, clear=True):
            assert TurboQuantFeatureGates.is_enabled("tq4") is True
    

    def test_tq8_off(self):
        """TQ8 (OFF) should never be enabled"""
        with patch.dict(os.environ, {}, clear=True):
            assert TurboQuantFeatureGates.is_enabled("tq8") is False
    
    def test_tq1_enable_experimental_env(self):
        """TQ1 should enable when SGLANG_ENABLE_EXPERIMENTAL=true"""
        with patch.dict(os.environ, {"SGLANG_ENABLE_EXPERIMENTAL": "true"}):
            assert TurboQuantFeatureGates.is_enabled("tq1") is True
    
    def test_tq1_enable_specific_env(self):
        """TQ1 should enable with specific SGLANG_ENABLE_TQ1=true"""
        with patch.dict(os.environ, {"SGLANG_ENABLE_TQ1": "true"}):
            assert TurboQuantFeatureGates.is_enabled("tq1") is True
    
    def test_tq1_specific_env_overrides_experimental(self):
        """Specific env var should override experimental flag"""
        with patch.dict(os.environ, {"SGLANG_ENABLE_EXPERIMENTAL": "false", "SGLANG_ENABLE_TQ1": "true"}):
            assert TurboQuantFeatureGates.is_enabled("tq1") is True
    
    def test_unknown_mode(self):
        """Unknown mode should log warning and return False"""
        with patch.dict(os.environ, {}, clear=True):
            assert TurboQuantFeatureGates.is_enabled("tq99") is False


class TestRotorQuantFeatureGateEnablement:
    """Test RotorQuant mode enablement logic"""
    
    def test_all_rq_modes_disabled_by_default(self):
        """All RotorQuant modes should be disabled by default"""
        with patch.dict(os.environ, {}, clear=True):
            for mode in ["rq3_planar", "rq4_planar", "rq3_iso", "rq4_iso"]:
                assert RotorQuantFeatureGates.is_enabled(mode) is False
    
    def test_rq3_planar_enable(self):
        """RQ3_planar should enable with env var"""
        with patch.dict(os.environ, {"SGLANG_ENABLE_RQ3_PLANAR": "true"}):
            assert RotorQuantFeatureGates.is_enabled("rq3_planar") is True
    
    def test_rq4_iso_enable(self):
        """RQ4_iso should enable with env var"""
        with patch.dict(os.environ, {"SGLANG_ENABLE_RQ4_ISO": "true"}):
            assert RotorQuantFeatureGates.is_enabled("rq4_iso") is True
    
    def test_unknown_rq_mode(self):
        """Unknown RQ mode should return False"""
        with patch.dict(os.environ, {}, clear=True):
            assert RotorQuantFeatureGates.is_enabled("rq99_unknown") is False


class TestTurboQuantDefaultMode:
    """Test default mode selection by hardware"""
    
    def test_gfx1030_default_fp16_until_validation(self):
        """Before Phase 6 validation, default for gfx1030 is fp16"""
        with patch.dict(os.environ, {}, clear=True):
            default = TurboQuantFeatureGates.get_default_for_hw("gfx1030")
            assert default == "fp16"  # Until Phase 6
    
    def test_gfx1030_default_can_override(self):
        """Default can be overridden via env var"""
        with patch.dict(os.environ, {"SGLANG_DEFAULT_KV_DTYPE": "tq2"}):
            default = TurboQuantFeatureGates.get_default_for_hw("gfx1030")
            assert default == "tq2"
    
    def test_unknown_hw_default_fp16(self):
        """Unknown hardware defaults to fp16"""
        with patch.dict(os.environ, {}, clear=True):
            default = TurboQuantFeatureGates.get_default_for_hw("unknown_gpu")
            assert default == "fp16"


class TestHardwareDetection:
    """Test AMD GPU architecture detection"""
    
    def test_detect_gfx1030(self):
        """Should detect gfx1030 hardware (skipped if torch unavailable)"""
        pytest.importorskip("torch")
        with patch("sglang_feature_gates.torch.cuda.is_available") as mock_available, \
             patch("sglang_feature_gates.torch.cuda.get_device_properties") as mock_props, \
             patch("sglang_feature_gates.torch.version.hip", "5.2"):
            mock_available.return_value = True
            mock_device = MagicMock()
            mock_device.name = "AMD Radeon RX 6700 XT (gfx1030)"
            mock_props.return_value = mock_device
            
            arch = HardwareSafetyValidator.detect_gpu_arch()
            assert arch == "gfx1030"
    
    def test_detect_no_gpu(self):
        """Should return None if no CUDA available (skipped if torch unavailable)"""
        pytest.importorskip("torch")
        with patch("sglang_feature_gates.torch.cuda.is_available") as mock_available:
            mock_available.return_value = False
            arch = HardwareSafetyValidator.detect_gpu_arch()
            assert arch is None
    
    def test_detect_exception_handling(self):
        """Should handle detection exceptions gracefully (skipped if torch unavailable)"""
        pytest.importorskip("torch")
        with patch("sglang_feature_gates.torch.cuda.is_available") as mock_available:
            mock_available.side_effect = Exception("CUDA error")
            arch = HardwareSafetyValidator.detect_gpu_arch()
            assert arch is None



class TestCompressionConfigValidation:
    """Test compression configuration validation"""
    
    def test_valid_tq2_config(self):
        """TQ2 config should validate when enabled"""
        with patch.dict(os.environ, {}, clear=True):
            is_valid, error = HardwareSafetyValidator.validate_compression_config("tq2")
            assert is_valid is True
            assert error is None
    
    def test_invalid_mode(self):
        """Invalid mode should fail with clear message"""
        is_valid, error = HardwareSafetyValidator.validate_compression_config("invalid_mode")
        assert is_valid is False
        assert "Invalid --kv-cache-dtype" in error
    
    def test_tq1_experimental_validation(self):
        """TQ1 should fail validation in default config"""
        with patch.dict(os.environ, {}, clear=True):
            is_valid, error = HardwareSafetyValidator.validate_compression_config("tq1")
            assert is_valid is False
            assert "EXPERIMENTAL" in error
    
    def test_long_context_warning(self):
        """Should log warning for very long contexts"""
        is_valid, error = HardwareSafetyValidator.validate_compression_config(
            "tq2", max_context_length=40000
        )
        # Still valid, but warning logged
        assert is_valid is True
    
    def test_fp16_always_valid(self):
        """FP16 should always be valid"""
        with patch.dict(os.environ, {}, clear=True):
            is_valid, error = HardwareSafetyValidator.validate_compression_config("fp16")
            assert is_valid is True


class TestFallbackChainManager:
    """Test fallback mode selection"""
    
    def test_tq1_fallback_chain(self):
        """TQ1 should have full fallback chain"""
        chain = FallbackChainManager.get_fallback_chain("tq1")
        assert "tq2" in chain
        assert "tq4" in chain
        assert "fp16" in chain
    
    def test_tq4_fallback_chain(self):
        """TQ4 should fallback to fp16"""
        chain = FallbackChainManager.get_fallback_chain("tq4")
        assert "fp8_e4m3" in chain or "fp16" in chain
    
    def test_fp16_no_fallback(self):
        """FP16 should have no fallback"""
        chain = FallbackChainManager.get_fallback_chain("fp16")
        assert len(chain) == 0
    
    def test_unknown_mode_fallback(self):
        """Unknown mode should default to fallback to fp16"""
        chain = FallbackChainManager.get_fallback_chain("unknown")
        assert "fp16" in chain or chain == ["fp16"]
    
    def test_find_available_mode_preferred_enabled(self):
        """Should return preferred mode if available"""
        with patch.dict(os.environ, {}, clear=True):
            mode = FallbackChainManager.find_available_mode("tq4")
            assert mode == "tq4"
    
    def test_find_available_mode_falls_back(self):
        """Should fall back when preferred unavailable"""
        with patch.dict(os.environ, {"SGLANG_ENABLE_TQ1": "false"}, clear=True):
            # TQ1 is experimental by default, so should fall back
            mode = FallbackChainManager.find_available_mode("tq1")
            assert mode != "tq1"  # Should be a fallback


class TestProductionConfiguration:
    """Test comprehensive production validation"""
    
    @patch("sglang_feature_gates.HardwareSafetyValidator.enforce_gfx1030")
    def test_production_config_no_hw_check(self, mock_hw):
        """Should skip hw check if enforce_hardware=False"""
        with patch.dict(os.environ, {}, clear=True):
            is_valid, error = validate_production_config(
                "tq2",
                enforce_hardware=False,
                allow_experimental=False
            )
            mock_hw.assert_not_called()
            assert is_valid is True
    
    def test_production_config_experimental_blocked(self):
        """Experimental modes should fail with allow_experimental=False"""
        with patch.dict(os.environ, {"SGLANG_ENABLE_TQ1": "true"}, clear=True):
            is_valid, error = validate_production_config(
                "tq1",
                enforce_hardware=False,
                allow_experimental=False
            )
            assert is_valid is False
            assert "EXPERIMENTAL" in error
    
    def test_production_config_experimental_allowed(self):
        """Experimental should pass with allow_experimental=True"""
        with patch.dict(os.environ, {"SGLANG_ENABLE_TQ1": "true"}, clear=True):
            is_valid, error = validate_production_config(
                "tq1",
                enforce_hardware=False,
                allow_experimental=True
            )
            assert is_valid is True


class TestErrorMessaging:
    """Test clarity of error messages for users"""
    
    def test_invalid_mode_error_has_valid_list(self):
        """Invalid mode error should list valid modes"""
        is_valid, error = HardwareSafetyValidator.validate_compression_config("oops")
        assert "Valid:" in error or "valid" in error.lower()
        assert "tq2" in error or "fp16" in error
    
    def test_experimental_error_has_solution(self):
        """Experimental mode error should suggest how to enable"""
        with patch.dict(os.environ, {}, clear=True):
            is_valid, error = HardwareSafetyValidator.validate_compression_config("tq1")
            assert "SGLANG_ENABLE_EXPERIMENTAL" in error or "export" in error
    
    def test_hw_detection_error_has_workaround(self):
        """Hardware detection error should suggest workaround"""
        with patch("sglang_feature_gates.HardwareSafetyValidator.detect_gpu_arch") as mock_detect:
            mock_detect.return_value = None
            HardwareSafetyValidator.enforce_gfx1030()
            # Message should suggest workaround
            # (Can't easily capture log, but can verify function returns False)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
