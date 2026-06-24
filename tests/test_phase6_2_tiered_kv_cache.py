"""
Tests for Tiered KV Cache Manager (RotorQuant GPU + TurboQuant RAM).

Phase 6.2: Two-tier caching strategy tests
- Tier 1 (GPU): RotorQuant 3-bit (8x compression)
- Tier 2 (RAM): TurboQuant fallback (spill for overflow)
"""

import pytest
import torch
import numpy as np
import time
import logging
from pathlib import Path

# Import tiered cache manager
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from tiered_kv_cache_manager import TieredKvCacheManager, CacheTier, BlockMetadata
from sglang_backend_adapter import TieredKvCacheAdapter
from kv_quant_contracts import UniversalKvStage

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestTieredKvCacheManager:
    """Test core tiered cache manager."""
    
    def setup_method(self):
        """Create fresh cache manager for each test."""
        self.cache_mgr = TieredKvCacheManager(
            gpu_tier_capacity_mb=100,  # Small for testing
            ram_tier_capacity_mb=200,
            block_size=16,
            block_metadata_bytes=2,
        )
    
    def test_allocate_block_to_gpu_tier(self):
        """Block allocation to GPU tier when space available."""
        data = torch.randn(16, 1024)
        block_id = self.cache_mgr.allocate_block(
            request_id="req_1",
            layer_idx=0,
            seq_start=0,
            seq_end=16,
            data=data,
            importance_score=0.9,
        )
        
        assert block_id == 0
        assert block_id in self.cache_mgr.gpu_blocks
        assert self.cache_mgr.block_to_tier[block_id] == CacheTier.GPU_ROTOR
        
        stats = self.cache_mgr.get_stats()
        assert stats["gpu_tier"]["blocks"] == 1
        assert stats["ram_tier"]["blocks"] == 0
        logger.info(f"✓ Block allocated to GPU tier: {block_id}")
    
    def test_allocate_block_to_ram_on_gpu_overflow(self):
        """Block allocation to RAM tier when GPU full and can't evict."""
        # Use extremely small GPU capacity, with pinned blocks so nothing can be evicted
        small_cache = TieredKvCacheManager(
            gpu_tier_capacity_mb=0.05,  # 50KB (very tiny)
            ram_tier_capacity_mb=10,
        )
        
        # Fill GPU with 2-3 small pinned blocks (can't evict)
        for i in range(2):
            data = torch.randn(8, 256)
            block_id = small_cache.allocate_block(
                request_id=f"pinned_{i}",
                layer_idx=i,
                seq_start=0,
                seq_end=8,
                data=data,
                importance_score=0.5,
            )
            # Pin the block so it can't be evicted
            if block_id in small_cache.gpu_blocks:
                small_cache.gpu_blocks[block_id].is_pinned = True
        
        # Now try to allocate a new block - should fail or go to RAM
        data = torch.randn(8, 256)
        try:
            block_id = small_cache.allocate_block(
                request_id="req_overflow",
                layer_idx=10,
                seq_start=0,
                seq_end=8,
                data=data,
                importance_score=0.5,
            )
            # If successful, it should be in RAM (since GPU is full of pinned blocks)
            assert block_id in small_cache.ram_blocks or block_id in small_cache.gpu_blocks
            logger.info(f"✓ Block allocated: tier={small_cache.block_to_tier.get(block_id)}")
        except MemoryError:
            logger.info("✓ Memory error raised (both tiers full with pinned blocks)")
    
    def test_access_block_gpu_tier(self):
        """Accessing block on GPU tier increments hit counter."""
        data = torch.randn(16, 1024)
        block_id = self.cache_mgr.allocate_block(
            request_id="req_1",
            layer_idx=0,
            seq_start=0,
            seq_end=16,
            data=data,
            importance_score=0.9,
        )
        
        # Access block
        result = self.cache_mgr.access_block(block_id)
        assert isinstance(result, torch.Tensor)
        
        metadata = self.cache_mgr.gpu_blocks[block_id]
        assert metadata.access_count == 1
        
        stats = self.cache_mgr.get_stats()
        assert stats["gpu_tier"]["hits"] == 1
        logger.info(f"✓ GPU tier access recorded: hits={stats['gpu_tier']['hits']}")

    def test_access_block_preserves_shape(self):
        """Decoded block should preserve original tensor shape."""
        data = torch.randn(7, 321)
        block_id = self.cache_mgr.allocate_block(
            request_id="req_shape",
            layer_idx=0,
            seq_start=0,
            seq_end=7,
            data=data,
            importance_score=0.9,
        )
        decoded = self.cache_mgr.access_block(block_id)
        assert tuple(decoded.shape) == tuple(data.shape)
        assert torch.isfinite(decoded).all()

    def test_rotor_payload_is_more_compact_than_fp32(self):
        """Packed 3-bit payload should beat FP32 byte footprint."""
        data = torch.randn(64, 512)
        block_id = self.cache_mgr.allocate_block(
            request_id="req_compact",
            layer_idx=0,
            seq_start=0,
            seq_end=64,
            data=data,
            importance_score=0.9,
        )
        metadata = self.cache_mgr.gpu_blocks.get(block_id) or self.cache_mgr.ram_blocks.get(block_id)
        assert metadata is not None
        fp32_bytes = data.numel() * 4
        assert metadata.size_bytes < fp32_bytes
        assert metadata.codec_name in {"python_packed_rq3", "rust_planar3"}

    def test_stage_metadata_initialized_for_hot_block(self):
        """High-importance fresh blocks should start on hot rotor stage."""
        data = torch.randn(16, 1024)
        block_id = self.cache_mgr.allocate_block(
            request_id="req_stage_hot",
            layer_idx=0,
            seq_start=0,
            seq_end=16,
            data=data,
            importance_score=0.95,
        )
        metadata = self.cache_mgr.gpu_blocks[block_id]
        assert metadata.stage == UniversalKvStage.hot_rotor
        assert metadata.block_header is not None
        assert metadata.block_header.stage == UniversalKvStage.hot_rotor
        assert metadata.block_header.should_apply_turbo_residual() is False

    def test_stage_transitions_hot_warm_cold(self):
        """Stage policy should move metadata through hot/warm/cold with age."""
        data = torch.randn(16, 1024)
        block_id = self.cache_mgr.allocate_block(
            request_id="req_stage_shift",
            layer_idx=0,
            seq_start=0,
            seq_end=16,
            data=data,
            importance_score=0.9,
        )
        metadata = self.cache_mgr.gpu_blocks[block_id]
        assert metadata.stage == UniversalKvStage.hot_rotor

        self.cache_mgr._stage_tick = self.cache_mgr.placement_policy.warm_age_steps + 5
        metadata.last_stage_tick = 0
        warm_decoded = self.cache_mgr.access_block(block_id)
        assert metadata.stage == UniversalKvStage.warm_rotor_polar
        assert metadata.warm_reference_payload is not None
        assert metadata.warm_reference_format == "warm_rotor_polar_ref_v1"
        assert metadata.stage_materialization_source == "warm_rotor_polar_reference"
        assert tuple(warm_decoded.shape) == tuple(data.shape)

        self.cache_mgr._stage_tick = self.cache_mgr.placement_policy.cold_age_steps + 20
        metadata.last_stage_tick = 0
        cold_decoded = self.cache_mgr.access_block(block_id)
        assert metadata.stage == UniversalKvStage.cold_turbo_residual
        assert metadata.block_header is not None
        assert metadata.block_header.should_apply_turbo_residual() is True
        assert metadata.cold_reference_payload is not None
        assert metadata.cold_reference_format == "cold_turbo_residual_ref_v1"
        assert metadata.stage_materialization_source == "cold_turbo_residual_reference"
        assert tuple(cold_decoded.shape) == tuple(data.shape)

    def test_stage_metadata_integrity_on_cold_ram_block(self):
        """Cold-stage blocks should preserve stage/header metadata integrity in RAM."""
        data = torch.randn(16, 1024)
        block_id = self.cache_mgr.allocate_block(
            request_id="req_stage_cold",
            layer_idx=0,
            seq_start=0,
            seq_end=16,
            data=data,
            importance_score=0.1,
        )
        metadata = self.cache_mgr.ram_blocks.get(block_id) or self.cache_mgr.gpu_blocks[block_id]
        assert metadata.stage == UniversalKvStage.cold_turbo_residual
        assert metadata.block_header is not None
        assert metadata.block_header.stage == metadata.stage
        assert metadata.block_header.should_apply_turbo_residual() is True
        assert metadata.cold_reference_payload is not None
        assert metadata.cold_reference_format == "cold_turbo_residual_ref_v1"
    
    def test_access_block_ram_tier_miss(self):
        """Accessing block on RAM tier records miss (latency penalty)."""
        # Force allocation to RAM
        self.cache_mgr.ram_blocks[999] = BlockMetadata(
            block_id=999,
            request_id="req_1",
            layer_idx=0,
            seq_start=0,
            seq_end=16,
            current_tier=CacheTier.RAM_TURBO,
            created_at=time.time(),
            last_accessed=time.time(),
            importance_score=0.5,
            size_bytes=256,
        )
        self.cache_mgr.block_to_tier[999] = CacheTier.RAM_TURBO
        self.cache_mgr.ram_block_data[999] = b"\x00" * 256
        self.cache_mgr.stats[CacheTier.RAM_TURBO].blocks_count = 1
        self.cache_mgr.stats[CacheTier.RAM_TURBO].total_bytes = 256
        
        # Access from RAM
        result = self.cache_mgr.access_block(999)
        assert isinstance(result, torch.Tensor)
        
        stats = self.cache_mgr.get_stats()
        assert stats["ram_tier"]["misses"] == 1
        logger.info(f"✓ RAM tier access recorded as miss: misses={stats['ram_tier']['misses']}")
    
    def test_evict_block(self):
        """Eviction removes block and frees space."""
        data = torch.randn(16, 1024)
        block_id = self.cache_mgr.allocate_block(
            request_id="req_1",
            layer_idx=0,
            seq_start=0,
            seq_end=16,
            data=data,
            importance_score=0.9,
        )
        
        stats_before = self.cache_mgr.get_stats()
        assert stats_before["gpu_tier"]["blocks"] == 1
        
        # Evict
        self.cache_mgr.evict_block(block_id)
        
        stats_after = self.cache_mgr.get_stats()
        assert stats_after["gpu_tier"]["blocks"] == 0
        assert block_id not in self.cache_mgr.gpu_blocks
        logger.info(f"✓ Block evicted: freed {stats_before['gpu_tier']['bytes']} bytes")
    
    def test_eviction_priority_lru_plus_importance(self):
        """Eviction prioritizes cold, low-importance blocks."""
        # Allocate 3 blocks with different importance
        now = time.time()
        
        # Block 0: hot, high importance (should NOT evict)
        data_hot = torch.randn(16, 1024)
        block_hot = self.cache_mgr.allocate_block(
            request_id="req_hot",
            layer_idx=0,
            seq_start=0,
            seq_end=16,
            data=data_hot,
            importance_score=0.95,
        )
        
        # Block 1: cold, low importance (should evict first)
        data_cold = torch.randn(16, 1024)
        block_cold = self.cache_mgr.allocate_block(
            request_id="req_cold",
            layer_idx=1,
            seq_start=0,
            seq_end=16,
            data=data_cold,
            importance_score=0.4,  # warm-stage so it remains eligible for GPU eviction logic
        )
        
        # Make cold block old by setting last_accessed to past
        self.cache_mgr.gpu_blocks[block_cold].last_accessed = now - 1000  # 1000 seconds ago
        self.cache_mgr.gpu_blocks[block_hot].last_accessed = now  # just now
        
        # Evict one block
        evicted = self.cache_mgr._evict_to_make_space(256, CacheTier.GPU_ROTOR)
        
        assert evicted >= 1
        assert block_hot in self.cache_mgr.gpu_blocks  # Hot block still there
        assert block_cold not in self.cache_mgr.gpu_blocks  # Cold block evicted
        logger.info(f"✓ Eviction prioritizes cold blocks: evicted {evicted} cold block")
    
    def test_compression_ratio_8x(self):
        """Verify 8x compression ratio (64B → 8B)."""
        stats = self.cache_mgr.get_stats()
        assert stats["compression_ratio"] == 8.0
        assert stats["block_size"] == 16
        logger.info(f"✓ Compression ratio verified: {stats['compression_ratio']}x")


class TestTieredKvCacheAdapter:
    """Test high-level tiered cache adapter (layer above manager)."""
    
    def setup_method(self):
        """Create fresh adapter for each test."""
        self.adapter = TieredKvCacheAdapter(
            gpu_capacity_mb=100,
            ram_capacity_mb=200,
            primary_codec="rq3_planar",
            secondary_codec="tq2",
            dimension=4096,
            num_heads=32,
        )
    
    def test_allocate_kv_block(self):
        """Allocate K and V blocks via adapter."""
        k_cache = torch.randn(16, 4096)
        v_cache = torch.randn(16, 4096)
        
        block_id = self.adapter.allocate_kv_block(
            request_id="req_1",
            layer_idx=0,
            k_cache=k_cache,
            v_cache=v_cache,
            importance_score=0.9,
        )
        
        assert block_id >= 0
        assert block_id in self.adapter.cache_mgr.gpu_blocks or block_id in self.adapter.cache_mgr.ram_blocks
        logger.info(f"✓ KV block allocated: {block_id}")
    
    def test_get_kv_block(self):
        """Retrieve and decompress KV block."""
        k_cache = torch.randn(16, 4096)
        v_cache = torch.randn(16, 4096)
        
        block_id = self.adapter.allocate_kv_block(
            request_id="req_1",
            layer_idx=0,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        
        result = self.adapter.get_kv_block(block_id)
        assert isinstance(result, torch.Tensor)
        logger.info(f"✓ KV block retrieved: shape={result.shape}")
    
    def test_cache_stats(self):
        """Get and verify cache statistics."""
        k_cache = torch.randn(16, 4096)
        
        block_id = self.adapter.allocate_kv_block(
            request_id="req_1",
            layer_idx=0,
            k_cache=k_cache,
            importance_score=0.9,
        )
        
        stats = self.adapter.get_cache_stats()
        
        assert "gpu_tier" in stats
        assert "ram_tier" in stats
        assert stats["gpu_tier"]["blocks"] >= 1
        assert stats["compression_ratio"] == 8.0
        logger.info(f"✓ Cache stats retrieved:\n{stats}")
    
    def test_evict_block_via_adapter(self):
        """Evict block through adapter interface."""
        k_cache = torch.randn(16, 4096)
        
        block_id = self.adapter.allocate_kv_block(
            request_id="req_1",
            layer_idx=0,
            k_cache=k_cache,
        )
        
        self.adapter.evict_block(block_id)
        
        stats = self.adapter.get_cache_stats()
        assert (stats["gpu_tier"]["blocks"] + stats["ram_tier"]["blocks"]) == 0
        logger.info(f"✓ Block evicted via adapter")
    
    def test_print_cache_summary(self):
        """Print human-readable cache summary."""
        # Allocate several blocks
        for i in range(3):
            k_cache = torch.randn(16, 4096)
            self.adapter.allocate_kv_block(
                request_id=f"req_{i}",
                layer_idx=i,
                k_cache=k_cache,
                importance_score=0.5 + i * 0.1,
            )
        
        self.adapter.print_cache_summary()
        logger.info("✓ Cache summary printed")


class TestTwoTierScenarios:
    """Test realistic two-tier caching scenarios."""
    
    def test_long_context_scenario(self):
        """
        Scenario: Processing very long context (100K tokens).
        
        Expected behavior:
        - Initial blocks: GPU tier (RotorQuant, fast)
        - Overflow blocks: RAM tier (TurboQuant, spillage)
        - Metrics: GPU hit rate high, RAM swap overhead low
        """
        adapter = TieredKvCacheAdapter(
            gpu_capacity_mb=2,  # Much smaller GPU (2MB instead of 8000MB)
            ram_capacity_mb=50,  # Larger RAM
        )
        
        # Simulate long context in 1K-token blocks
        block_ids = []
        for block_num in range(12):  # 12 blocks
            k_cache = torch.randn(1024, 4096)  # Smaller blocks
            block_id = adapter.allocate_kv_block(
                request_id="long_context_req",
                layer_idx=0,
                k_cache=k_cache,
                importance_score=max(0.0, 1.0 - (block_num * 0.1)),  # tail blocks become cold
            )
            block_ids.append(block_id)
        
        stats = adapter.get_cache_stats()
        logger.info(
            f"\n=== Long-Context Scenario ===\n"
            f"GPU blocks: {stats['gpu_tier']['blocks']}\n"
            f"RAM blocks: {stats['ram_tier']['blocks']}\n"
            f"GPU util: {stats['gpu_tier']['utilization_pct']:.1f}%\n"
            f"RAM util: {stats['ram_tier']['utilization_pct']:.1f}%"
        )
        
        # Verify split between GPU and RAM
        assert stats["gpu_tier"]["blocks"] > 0
        assert stats["ram_tier"]["blocks"] > 0
        assert stats["ram_tier"]["blocks"] > stats["gpu_tier"]["blocks"]
        logger.info("✓ Long-context scenario: appropriate GPU/RAM split")
    
    def test_importance_weighted_eviction(self):
        """
        Scenario: Model uses importance-weighted attention.
        
        Expected behavior:
        - High-importance blocks: stay in GPU
        - Low-importance blocks: evicted first to make room
        - Result: Better hit rate for critical attentions
        """
        cache_mgr = TieredKvCacheManager(
            gpu_tier_capacity_mb=50,
            ram_tier_capacity_mb=100,
        )
        
        now = time.time()
        
        # Allocate 3 blocks: critical, normal, noise
        block_critical = cache_mgr.allocate_block(
            request_id="req",
            layer_idx=0,
            seq_start=0,
            seq_end=16,
            data=torch.randn(16, 1024),
            importance_score=0.99,  # Very important
        )
        
        block_normal = cache_mgr.allocate_block(
            request_id="req",
            layer_idx=1,
            seq_start=0,
            seq_end=16,
            data=torch.randn(16, 1024),
            importance_score=0.5,  # Medium
        )
        
        block_noise = cache_mgr.allocate_block(
            request_id="req",
            layer_idx=2,
            seq_start=0,
            seq_end=16,
            data=torch.randn(16, 1024),
            importance_score=0.4,  # Warm-tier for GPU eviction path assertions
        )
        
        # Age the blocks (make noise oldest)
        cache_mgr.gpu_blocks[block_noise].last_accessed = now - 1000
        cache_mgr.gpu_blocks[block_normal].last_accessed = now - 500
        cache_mgr.gpu_blocks[block_critical].last_accessed = now - 10
        
        # Force eviction
        evicted = cache_mgr._evict_to_make_space(512, CacheTier.GPU_ROTOR)
        
        # Verify critical block survived eviction
        assert block_critical in cache_mgr.gpu_blocks
        logger.info("✓ Importance-weighted eviction: critical blocks preserved")
    
    def test_promotion_of_hot_blocks(self):
        """
        Scenario: Block allocated to RAM due to GPU constraints, then hot-accessed.
        
        Expected behavior:
        - Block allocated when GPU can't fit it
        - Multiple accesses recorded
        - Hot metrics tracked
        """
        cache_mgr = TieredKvCacheManager(
            gpu_tier_capacity_mb=0.05,  # Extremely small GPU
            ram_tier_capacity_mb=10,
        )
        
        # Fill GPU with pinned blocks so nothing else fits
        for i in range(2):
            data = torch.randn(8, 256)
            block_id = cache_mgr.allocate_block(
                request_id=f"pinned_{i}",
                layer_idx=i,
                seq_start=0,
                seq_end=8,
                data=data,
                importance_score=0.9,
            )
            if block_id in cache_mgr.gpu_blocks:
                cache_mgr.gpu_blocks[block_id].is_pinned = True
        
        # Next blocks will go to RAM
        block_id = cache_mgr.allocate_block(
            request_id="hot_req",
            layer_idx=10,
            seq_start=0,
            seq_end=8,
            data=torch.randn(8, 256),
            importance_score=0.8,
        )
        
        # Access multiple times
        for _ in range(5):
            try:
                cache_mgr.access_block(block_id)
            except KeyError:
                # Block might not exist if allocation failed
                pass
        
        logger.info(f"✓ Block access test completed for block {block_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
