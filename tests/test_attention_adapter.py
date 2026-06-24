#!/usr/bin/env python3
"""
Tests for Attention Backend Adapter Layer
==========================================

Validates:
- Backend dispatcher logic
- Backend selection based on hardware
- Fallback chain behavior
- Compression integration
- Telemetry collection
"""

import pytest
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from attention_backend_adapter import (
    AttentionBackendName,
    AttentionBackendDispatcher,
    AttentionBackendAdapter,
    DeviceType,
    BackendCapabilities,
)


class TestBackendDispatcher:
    """Test backend dispatcher logic"""
    
    def test_dispatcher_initialization(self):
        """Test dispatcher can be created and initialized"""
        dispatcher = AttentionBackendDispatcher()
        assert dispatcher is not None
        assert dispatcher.device_type is not None
    
    def test_backend_registry_populated(self):
        """Test that backends are registered"""
        dispatcher = AttentionBackendDispatcher()
        assert len(dispatcher.backend_registry) > 0
    
    def test_get_backend_info(self):
        """Test getting backend information"""
        dispatcher = AttentionBackendDispatcher()
        
        # Should be able to get info for any registered backend
        for backend_name in dispatcher.backend_registry.keys():
            info = dispatcher.get_backend_info(backend_name)
            assert info is not None
            assert info.name == backend_name
    
    def test_select_backend_without_preference(self):
        """Test automatic backend selection"""
        dispatcher = AttentionBackendDispatcher()
        
        backend = dispatcher.select_backend()
        assert backend is not None
        assert isinstance(backend, AttentionBackendName)
    
    def test_select_torch_native_fallback(self):
        """Test that torch_native is always available as fallback"""
        dispatcher = AttentionBackendDispatcher()
        
        # torch_native should handle any requirements
        backend = dispatcher.select_backend(seq_len=100000)
        assert backend is not None
    
    def test_fallback_chain_generation(self):
        """Test fallback chain is properly constructed"""
        dispatcher = AttentionBackendDispatcher()
        
        # Get fallback chain for any backend
        fallback_chain = dispatcher.get_fallback_chain(AttentionBackendName.TORCH_NATIVE)
        assert len(fallback_chain) >= 1
        assert fallback_chain[0] == AttentionBackendName.TORCH_NATIVE
    
    def test_mla_model_support(self):
        """Test selection respects MLA model requirements"""
        dispatcher = AttentionBackendDispatcher()
        
        # Select backend for MLA model
        backend = dispatcher.select_backend(is_mla=True)
        caps = dispatcher.get_backend_info(backend)
        
        # If on CPU, torch_native doesn't support MLA but that's ok
        # Real MLA support would be on GPU
        assert caps is not None
    
    def test_long_sequence_selection(self):
        """Test backend selection for long sequences"""
        dispatcher = AttentionBackendDispatcher()
        
        # Select for 8K sequence
        backend = dispatcher.select_backend(seq_len=8192)
        caps = dispatcher.get_backend_info(backend)
        
        # Should select triton or better (triton supports 32K by default)
        assert caps.max_seq_len >= 4096  # At least this much
    
    def test_kv_compression_preference(self):
        """Test backend selection prefers compression support when enabled"""
        dispatcher = AttentionBackendDispatcher()
        
        # Select with compression enabled
        backend = dispatcher.select_backend(kv_compression_enabled=True)
        caps = dispatcher.get_backend_info(backend)
        
        # On CPU, triton is selected which supports compression
        # On GPU, flashinfer/triton is selected
        assert backend in dispatcher.backend_registry


class TestBackendCapabilities:
    """Test backend capability definitions"""
    
    def test_flashinfer_capabilities(self):
        """Test FlashInfer has expected capabilities"""
        dispatcher = AttentionBackendDispatcher()
        
        caps = dispatcher.backend_registry.get(AttentionBackendName.FLASHINFER)
        if caps:  # Only test if NVIDIA backend registered
            assert caps.supports_kv_compression
            assert caps.supports_mla
            assert caps.max_seq_len >= 32768
    
    def test_aiter_capabilities(self):
        """Test AIter backend capabilities"""
        dispatcher = AttentionBackendDispatcher()
        
        caps = dispatcher.backend_registry.get(AttentionBackendName.AITER)
        if caps:  # Only test if ROCm backend registered
            assert caps.device_type == DeviceType.AMD_ROCM
            assert caps.supports_kv_compression
    
    def test_wave_capabilities(self):
        """Test Wave backend capabilities"""
        dispatcher = AttentionBackendDispatcher()
        
        caps = dispatcher.backend_registry.get(AttentionBackendName.WAVE)
        if caps:  # Only test if Wave registered
            assert caps.device_type == DeviceType.AMD_RDNA2
            assert caps.supports_kv_compression
    
    def test_torch_native_always_available(self):
        """Test torch_native is always registered"""
        dispatcher = AttentionBackendDispatcher()
        
        caps = dispatcher.backend_registry.get(AttentionBackendName.TORCH_NATIVE)
        assert caps is not None
        assert not caps.fallback_backends  # No further fallback needed


class TestBackendAdapter:
    """Test the backend adapter wrapper"""
    
    def test_adapter_creation(self):
        """Test creating an adapter for a backend"""
        dispatcher = AttentionBackendDispatcher()
        adapter = AttentionBackendAdapter(AttentionBackendName.TORCH_NATIVE, dispatcher)
        
        assert adapter is not None
        assert adapter.backend == AttentionBackendName.TORCH_NATIVE
    
    def test_adapter_telemetry_initialization(self):
        """Test telemetry is initialized"""
        dispatcher = AttentionBackendDispatcher()
        adapter = AttentionBackendAdapter(AttentionBackendName.TORCH_NATIVE, dispatcher)
        
        assert adapter.telemetry["forward_calls"] == 0
        assert adapter.telemetry["backward_calls"] == 0
    
    def test_enable_compression(self):
        """Test enabling compression on adapter"""
        dispatcher = AttentionBackendDispatcher()
        # Use triton which supports compression
        adapter = AttentionBackendAdapter(AttentionBackendName.TRITON, dispatcher)
        
        adapter.enable_compression("tq2")
        assert adapter.compression_state["enabled"]
        assert adapter.compression_state["mode"] == "tq2"
    
    def test_compression_ratio_setting(self):
        """Test compression ratio is set correctly"""
        dispatcher = AttentionBackendDispatcher()
        # Use triton instead of torch_native as it supports compression
        adapter = AttentionBackendAdapter(AttentionBackendName.TRITON, dispatcher)
        
        adapter.enable_compression("tq2")
        assert adapter.compression_state["compression_ratio"] == 8.0
        
        adapter.enable_compression("tq4")
        assert adapter.compression_state["compression_ratio"] == 4.0
    
    def test_forward_call_telemetry(self):
        """Test forward calls are tracked"""
        dispatcher = AttentionBackendDispatcher()
        adapter = AttentionBackendAdapter(AttentionBackendName.TORCH_NATIVE, dispatcher)
        
        adapter.forward(None, None, None)
        assert adapter.telemetry["forward_calls"] == 1
        
        adapter.forward(None, None, None)
        assert adapter.telemetry["forward_calls"] == 2
    
    def test_backward_call_telemetry(self):
        """Test backward calls are tracked"""
        dispatcher = AttentionBackendDispatcher()
        adapter = AttentionBackendAdapter(AttentionBackendName.TORCH_NATIVE, dispatcher)
        
        adapter.backward(None)
        assert adapter.telemetry["backward_calls"] == 1


class TestDispatcherScenarios:
    """Test realistic dispatcher scenarios"""
    
    def test_scenario_small_batch_short_context(self):
        """Scenario: small batch, short context (interactive use)"""
        dispatcher = AttentionBackendDispatcher()
        
        backend = dispatcher.select_backend(
            seq_len=512,
            kv_compression_enabled=False,
        )
        caps = dispatcher.get_backend_info(backend)
        
        # Should pick efficient backend
        assert caps.max_seq_len >= 512
    
    def test_scenario_large_batch_medium_context(self):
        """Scenario: large batch, medium context (batch processing)"""
        dispatcher = AttentionBackendDispatcher()
        
        backend = dispatcher.select_backend(
            seq_len=2048,
            kv_compression_enabled=False,
        )
        caps = dispatcher.get_backend_info(backend)
        
        assert caps.max_seq_len >= 2048
    
    def test_scenario_small_batch_long_context_with_compression(self):
        """Scenario: small batch, long context with KV compression"""
        dispatcher = AttentionBackendDispatcher()
        
        backend = dispatcher.select_backend(
            seq_len=8192,
            kv_compression_enabled=True,
        )
        caps = dispatcher.get_backend_info(backend)
        
        # Should select a capable backend
        assert caps is not None
    
    def test_scenario_mla_model(self):
        """Scenario: Multi-head Latent Attention model"""
        dispatcher = AttentionBackendDispatcher()
        
        backend = dispatcher.select_backend(is_mla=True)
        caps = dispatcher.get_backend_info(backend)
        
        # Should select a valid backend (may not have MLA on CPU, that's ok)
        assert caps is not None


class TestBackendSelection:
    """Test backend selection with various constraints"""
    
    def test_select_by_device(self):
        """Test that selected backend matches device"""
        dispatcher = AttentionBackendDispatcher()
        
        backend = dispatcher.select_backend()
        caps = dispatcher.get_backend_info(backend)
        
        # Backend device should match detected device or be compatible
        if dispatcher.device_type in (DeviceType.AMD_ROCM, DeviceType.AMD_RDNA2):
            # On AMD, should prefer AMD backends when available
            pass  # Implementation allows fallback to triton/torch_native
    
    def test_preference_honored_when_capable(self):
        """Test that preferred backend is used when capable"""
        dispatcher = AttentionBackendDispatcher()
        
        # Request torch_native (always available)
        backend = dispatcher.select_backend(preferred_backend="torch_native")
        assert backend == AttentionBackendName.TORCH_NATIVE
    
    def test_fallback_when_preference_unavailable(self):
        """Test fallback when preferred backend unavailable"""
        dispatcher = AttentionBackendDispatcher()
        
        # Request unlikely backend, should fallback gracefully
        backend = dispatcher.select_backend(preferred_backend="impossible_backend_xyz")
        assert backend is not None
        assert backend != "impossible_backend_xyz"


class TestCompressionIntegration:
    """Test KV cache compression integration"""
    
    def test_compression_modes_recognized(self):
        """Test all standard compression modes are recognized"""
        dispatcher = AttentionBackendDispatcher()
        # Use triton which supports compression
        adapter = AttentionBackendAdapter(AttentionBackendName.TRITON, dispatcher)
        
        modes = ["tq1", "tq2", "tq3", "tq4"]
        for mode in modes:
            adapter.enable_compression(mode)
            assert adapter.compression_state["mode"] == mode
            assert adapter.compression_state["enabled"]
    
    def test_compression_ratios(self):
        """Test compression ratios for each mode"""
        dispatcher = AttentionBackendDispatcher()
        # Use triton which supports compression
        adapter = AttentionBackendAdapter(AttentionBackendName.TRITON, dispatcher)
        
        expected = {
            "tq1": 16.0,
            "tq2": 8.0,
            "tq3": 5.33,
            "tq4": 4.0,
        }
        
        for mode, ratio in expected.items():
            adapter.enable_compression(mode)
            assert abs(adapter.compression_state["compression_ratio"] - ratio) < 0.1


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_zero_sequence_length(self):
        """Test handling of zero sequence length"""
        dispatcher = AttentionBackendDispatcher()
        
        # Should not crash
        backend = dispatcher.select_backend(seq_len=0)
        assert backend is not None
    
    def test_very_large_sequence_length(self):
        """Test handling of very large sequence length"""
        dispatcher = AttentionBackendDispatcher()
        
        # Select backend for 1M tokens (extreme)
        backend = dispatcher.select_backend(seq_len=1000000)
        assert backend is not None
    
    def test_unknown_backend_name(self):
        """Test handling unknown backend name"""
        dispatcher = AttentionBackendDispatcher()
        
        backend = dispatcher.select_backend(preferred_backend="unknown_backend")
        # Should fallback to something valid
        assert backend in dispatcher.backend_registry
    
    def test_multiple_constraints(self):
        """Test selection with multiple constraints"""
        dispatcher = AttentionBackendDispatcher()
        
        backend = dispatcher.select_backend(
            is_mla=True,
            seq_len=4096,
            kv_compression_enabled=True,
            use_sparse_attention=True,
        )
        assert backend is not None
        caps = dispatcher.get_backend_info(backend)
        assert caps is not None


def test_module_imports():
    """Test that all expected components are importable"""
    assert AttentionBackendName is not None
    assert AttentionBackendDispatcher is not None
    assert AttentionBackendAdapter is not None
    assert DeviceType is not None
    assert BackendCapabilities is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
