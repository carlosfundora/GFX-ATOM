"""
Tests for SGLang KV compression hooks.

Validates:
  - Phase 4.4.1: Encode hook after prefill
  - Phase 4.4.2: Decode hook before decode
  - Phase 4.4.3: KV pool allocator
  - Phase 4.4.4: Telemetry collection
"""

import pytest
import torch
from sglang_kv_compression import (
    CompressionMode,
    CompressionStatus,
    CompressedKV,
    TurboQuantBackend,
    RotorQuantBackend,
    KVCompressionManager,
    init_kv_compression,
    get_kv_compression_manager,
    compress_kv_if_enabled,
    decompress_kv_if_needed,
)


class TestTurboQuantBackend:
    """Test TurboQuantBackend encode/decode (Phase 4.4.1-4.4.2 STUBS)"""

    def test_tq2_encode_decode_roundtrip(self):
        """TQ2: Encode and decode should produce finite values"""
        backend = TurboQuantBackend(bit_width=2)
        
        # Create random KV tensor: [num_tokens, heads, dim]
        k_orig = torch.randn(64, 8, 128, dtype=torch.float32)
        
        # Encode
        k_compressed = backend.encode(k_orig)
        assert k_compressed.data is not None
        assert k_compressed.mode == CompressionMode.TQ2
        assert k_compressed.compression_ratio == 0.25
        
        # Decode
        k_decoded = backend.decode(k_compressed)
        assert k_decoded.shape == k_orig.shape
        assert torch.isfinite(k_decoded).all()

    def test_tq3_encode_decode_roundtrip(self):
        """TQ3: Check compression ratio is correct"""
        backend = TurboQuantBackend(bit_width=3)
        k_orig = torch.randn(32, 4, 64, dtype=torch.float32)
        
        k_compressed = backend.encode(k_orig)
        assert k_compressed.compression_ratio == 0.375
        
        k_decoded = backend.decode(k_compressed)
        assert torch.isfinite(k_decoded).all()

    def test_tq4_encode_decode_roundtrip(self):
        """TQ4: Compression ratio check"""
        backend = TurboQuantBackend(bit_width=4)
        k_orig = torch.randn(128, 16, 256, dtype=torch.float32)
        
        k_compressed = backend.encode(k_orig)
        assert k_compressed.compression_ratio == 0.5
        
        k_decoded = backend.decode(k_compressed)
        assert torch.isfinite(k_decoded).all()

    def test_tq1_experimental_mode(self):
        """TQ1: Experimental mode (note: only stubs supported in Phase 4.4)"""
        backend = TurboQuantBackend(bit_width=1)
        k_orig = torch.randn(32, 8, 128, dtype=torch.float32)
        
        k_compressed = backend.encode(k_orig)
        assert k_compressed.mode == CompressionMode.TQ1
        assert k_compressed.compression_ratio == 0.125

    def test_approximate_inner_product(self):
        """Approximate IP estimation (stub: should produce finite scores)"""
        backend = TurboQuantBackend(bit_width=2)
        
        # q: [seq_len, heads, dim] (not batched in SGLang during attention)
        # k: [seq_len, heads, dim]
        q = torch.randn(16, 8, 128, dtype=torch.float32)
        k_orig = torch.randn(64, 8, 128, dtype=torch.float32)
        
        k_compressed = backend.encode(k_orig)
        scores = backend.estimate_inner_product(q, k_compressed)
        
        # Result should be [seq_len_q, seq_len_k]
        assert scores.shape == (16, 64)
        assert torch.isfinite(scores).all()


class TestRotorQuantBackend:
    """Test RotorQuantBackend (future use, stubs for now)"""

    def test_rq3_planar_encode_decode(self):
        """RQ3 Planar mode encode/decode"""
        backend = RotorQuantBackend(bit_width=3, mode="planar")
        k_orig = torch.randn(64, 8, 128, dtype=torch.float32)
        
        k_compressed = backend.encode(k_orig)
        assert k_compressed.mode == CompressionMode.RQ3_PLANAR
        assert k_compressed.compression_ratio == 0.375
        
        k_decoded = backend.decode(k_compressed)
        assert torch.isfinite(k_decoded).all()

    def test_rq4_iso_encode_decode(self):
        """RQ4 Isometric mode encode/decode"""
        backend = RotorQuantBackend(bit_width=4, mode="iso")
        k_orig = torch.randn(32, 4, 64, dtype=torch.float32)
        
        k_compressed = backend.encode(k_orig)
        assert k_compressed.mode == CompressionMode.RQ4_ISO
        
        k_decoded = backend.decode(k_compressed)
        assert torch.isfinite(k_decoded).all()


class TestKVCompressionManager:
    """Test KVCompressionManager (Phase 4.4.3-4.4.4)"""

    def test_manager_init_tq2(self):
        """Initialize manager with TQ2"""
        manager = KVCompressionManager(mode=CompressionMode.TQ2)
        assert manager.backend is not None
        assert isinstance(manager.backend, TurboQuantBackend)

    def test_manager_init_fp16(self):
        """FP16 mode: no compression backend"""
        manager = KVCompressionManager(mode=CompressionMode.FP16)
        assert manager.backend is None

    def test_encode_kv_roundtrip(self):
        """Encode K and V from prefill (Phase 4.4.1 hook)"""
        manager = KVCompressionManager(mode=CompressionMode.TQ2, enable_telemetry=False)
        
        k_orig = torch.randn(64, 8, 128, dtype=torch.float32)
        v_orig = torch.randn(64, 8, 128, dtype=torch.float32)
        
        k_compressed, v_compressed = manager.encode_kv(k_orig, v_orig, layer_id=0)
        
        assert k_compressed.mode == CompressionMode.TQ2
        assert v_compressed.mode == CompressionMode.TQ2
        assert torch.isfinite(k_compressed.data).all()
        assert torch.isfinite(v_compressed.data).all()

    def test_decode_kv_roundtrip(self):
        """Decode K and V for decode attention (Phase 4.4.2 hook)"""
        manager = KVCompressionManager(mode=CompressionMode.TQ2, enable_telemetry=False)
        
        k_orig = torch.randn(64, 8, 128, dtype=torch.float32)
        v_orig = torch.randn(64, 8, 128, dtype=torch.float32)
        
        k_compressed, v_compressed = manager.encode_kv(k_orig, v_orig, layer_id=0)
        k_decoded, v_decoded = manager.decode_kv(k_compressed, v_compressed, layer_id=0)
        
        assert k_decoded.shape == k_orig.shape
        assert v_decoded.shape == v_orig.shape
        assert torch.isfinite(k_decoded).all()
        assert torch.isfinite(v_decoded).all()

    def test_fallback_chain_graceful_degradation(self):
        """Fallback chain: tq2 → tq3 → fp16"""
        manager = KVCompressionManager(mode=CompressionMode.TQ2, enable_telemetry=False)
        
        k = torch.randn(32, 4, 64, dtype=torch.float32)
        v = torch.randn(32, 4, 64, dtype=torch.float32)
        
        # Even with errors, should produce valid output via fallback
        k_compressed, v_compressed = manager.encode_kv(k, v, layer_id=0)
        k_decoded, v_decoded = manager.decode_kv(k_compressed, v_compressed, layer_id=0)
        
        assert torch.isfinite(k_decoded).all()
        assert torch.isfinite(v_decoded).all()

    def test_attention_score_estimation(self):
        """Estimate attention scores without full decompression"""
        manager = KVCompressionManager(mode=CompressionMode.TQ2, enable_telemetry=False)
        
        # q: [seq_len, heads, dim]
        # k: [seq_len, heads, dim]
        q = torch.randn(16, 8, 128, dtype=torch.float32)
        k_orig = torch.randn(64, 8, 128, dtype=torch.float32)
        
        k_compressed = manager.backend.encode(k_orig)
        scores = manager.estimate_attention_scores(q, k_compressed)
        
        assert scores.shape == (16, 64)
        assert torch.isfinite(scores).all()

    def test_telemetry_collection(self):
        """Collect and aggregate telemetry (Phase 4.4.4)"""
        manager = KVCompressionManager(
            mode=CompressionMode.TQ2,
            enable_telemetry=True  # Enable telemetry collection
        )
        
        for layer_id in range(4):
            k = torch.randn(64, 8, 128, dtype=torch.float32)
            v = torch.randn(64, 8, 128, dtype=torch.float32)
            manager.encode_kv(k, v, layer_id)
        
        stats = manager.get_stats()
        assert stats["mode"] == "tq2"
        assert stats["events"] == 4  # Should have 4 encode events


class TestGlobalCompressionManager:
    """Test global initialization and accessors"""

    def test_init_kv_compression_tq2(self):
        """Initialize global compression manager"""
        manager = init_kv_compression(mode="tq2", enable_telemetry=False)
        assert manager is not None
        assert get_kv_compression_manager() is manager

    def test_init_kv_compression_with_fallback(self):
        """Initialize with custom fallback chain"""
        manager = init_kv_compression(
            mode="tq2",
            fallback_chain=["tq3", "tq4", "fp16"],
            enable_telemetry=False
        )
        assert len(manager.fallback_chain) == 3

    def test_compress_kv_if_enabled_tq2(self):
        """Wrapper: compress when enabled"""
        init_kv_compression(mode="tq2", enable_telemetry=False)
        
        k = torch.randn(64, 8, 128, dtype=torch.float32)
        v = torch.randn(64, 8, 128, dtype=torch.float32)
        
        k_compressed, v_compressed = compress_kv_if_enabled(k, v, layer_id=0)
        assert k_compressed.mode == CompressionMode.TQ2
        assert v_compressed.mode == CompressionMode.TQ2

    def test_decompress_kv_if_needed(self):
        """Wrapper: decompress when needed"""
        init_kv_compression(mode="tq2", enable_telemetry=False)
        
        k = torch.randn(64, 8, 128, dtype=torch.float32)
        v = torch.randn(64, 8, 128, dtype=torch.float32)
        
        k_compressed, v_compressed = compress_kv_if_enabled(k, v, layer_id=0)
        k_decoded, v_decoded = decompress_kv_if_needed(k_compressed, v_compressed, layer_id=0)
        
        assert torch.isfinite(k_decoded).all()
        assert torch.isfinite(v_decoded).all()


class TestCompressedKVDataclass:
    """Test CompressedKV data container"""

    def test_size_bytes_without_metadata(self):
        """Calculate size with just data"""
        data = torch.randn(64, 8, 128, dtype=torch.float32)
        compressed = CompressedKV(data=data, mode=CompressionMode.TQ2)
        
        expected_bytes = 64 * 8 * 128 * 4  # float32 = 4 bytes
        assert compressed.size_bytes() == expected_bytes

    def test_size_bytes_with_scale(self):
        """Calculate size with scale metadata"""
        data = torch.randn(64, 8, 128, dtype=torch.float16)
        scale = torch.ones(8, dtype=torch.float32)
        compressed = CompressedKV(data=data, scale=scale, mode=CompressionMode.FP8_E4M3)
        
        data_bytes = 64 * 8 * 128 * 2  # float16 = 2 bytes
        scale_bytes = 8 * 4  # float32 = 4 bytes
        assert compressed.size_bytes() == data_bytes + scale_bytes


class TestCompressionModeEnum:
    """Test CompressionMode enum"""

    def test_all_modes_valid(self):
        """All defined modes should be accessible"""
        modes = [
            CompressionMode.FP16,
            CompressionMode.FP8_E4M3,
            CompressionMode.FP8_E5M2,
            CompressionMode.INT8,
            CompressionMode.INT4,
            CompressionMode.TQ1,
            CompressionMode.TQ2,
            CompressionMode.TQ3,
            CompressionMode.TQ4,
            CompressionMode.TQ8,
            CompressionMode.RQ3_PLANAR,
            CompressionMode.RQ4_PLANAR,
            CompressionMode.RQ3_ISO,
            CompressionMode.RQ4_ISO,
        ]
        assert len(modes) == 14

    def test_mode_from_string(self):
        """Create mode from string representation"""
        mode = CompressionMode("tq2")
        assert mode == CompressionMode.TQ2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
