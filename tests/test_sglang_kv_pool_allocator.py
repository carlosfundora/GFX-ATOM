"""
Tests for SGLang KV pool allocator (Phase 4.4.3).

Validates:
  - Compressed block allocation and deallocation
  - Memory usage tracking
  - Multi-layer coordination
  - Fragmentation and defragmentation
  - Page boundary handling
"""

import pytest
import torch
from sglang_kv_pool_allocator import (
    AllocationStrategy,
    AllocationStatistics,
    CompressedKVBlock,
    CompressedKVPool,
    MultiLayerCompressedKVAllocator,
    init_compressed_kv_allocator,
    get_compressed_kv_allocator,
    dtype_bytes,
)


class TestCompressedKVBlock:
    """Test CompressedKVBlock container"""

    def test_block_creation(self):
        """Create a compressed KV block"""
        data = torch.randn(64 * 8 * 32, dtype=torch.float16)  # Compressed data
        block = CompressedKVBlock(
            block_id=0,
            token_start=0,
            token_count=64,
            data=data,
            compression_mode="tq2",
        )
        
        assert block.block_id == 0
        assert block.token_count == 64
        assert block.size_bytes() == data.numel() * 2  # float16 = 2 bytes

    def test_block_with_metadata(self):
        """Block with scale and zero_point metadata"""
        data = torch.randn(64 * 8 * 32, dtype=torch.float16)
        scale = torch.ones(8, dtype=torch.float32)
        zero_point = torch.zeros(8, dtype=torch.float32)
        
        block = CompressedKVBlock(
            block_id=1,
            token_start=64,
            token_count=64,
            data=data,
            scale=scale,
            zero_point=zero_point,
            compression_mode="tq2",
        )
        
        expected_bytes = (data.numel() * 2) + (scale.numel() * 4) + (zero_point.numel() * 4)
        assert block.size_bytes() == expected_bytes

    def test_block_is_full(self):
        """Check if block reached page size"""
        data = torch.randn(64 * 8 * 32, dtype=torch.float16)
        block = CompressedKVBlock(
            block_id=0,
            token_start=0,
            token_count=16,
            data=data,
        )
        
        # Block with 16 tokens: full at page_size=16, not full at 17
        assert block.is_full(page_size=16)
        assert not block.is_full(page_size=17)


class TestCompressedKVPool:
    """Test single-layer KV pool"""

    def test_pool_allocation(self):
        """Allocate a block in the pool"""
        pool = CompressedKVPool(
            num_heads=8,
            head_dim=128,
            page_size=16,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
        )
        
        compressed_data = torch.randn(64 * 8 * 32, dtype=torch.float16)
        block = pool.allocate_block(
            token_start=0,
            token_count=64,
            compressed_data=compressed_data,
            compression_mode="tq2",
        )
        
        assert block.block_id == 0
        assert block.token_count == 64
        assert pool.stats.num_allocations == 1

    def test_pool_multiple_allocations(self):
        """Allocate multiple blocks"""
        pool = CompressedKVPool(
            num_heads=8,
            head_dim=128,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
        )
        
        for i in range(5):
            data = torch.randn(64 * 8 * 32, dtype=torch.float16)
            block = pool.allocate_block(
                token_start=i * 64,
                token_count=64,
                compressed_data=data,
            )
            assert block.block_id == i
        
        assert pool.stats.num_allocations == 5
        assert len(pool.blocks) == 5

    def test_pool_release_block(self):
        """Release a block"""
        pool = CompressedKVPool(
            num_heads=8,
            head_dim=128,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
        )
        
        data = torch.randn(64 * 8 * 32, dtype=torch.float16)
        block = pool.allocate_block(
            token_start=0,
            token_count=64,
            compressed_data=data,
        )
        
        assert pool.release_block(block.block_id)
        assert block.block_id not in pool.blocks
        assert pool.stats.num_deallocations == 1

    def test_pool_get_block(self):
        """Retrieve a block by ID"""
        pool = CompressedKVPool(
            num_heads=8,
            head_dim=128,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
        )
        
        data = torch.randn(64 * 8 * 32, dtype=torch.float16)
        original_block = pool.allocate_block(
            token_start=0,
            token_count=64,
            compressed_data=data,
        )
        
        retrieved = pool.get_block(original_block.block_id)
        assert retrieved is original_block
        assert retrieved.token_count == 64

    def test_pool_memory_usage(self):
        """Track memory usage"""
        pool = CompressedKVPool(
            num_heads=8,
            head_dim=128,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
        )
        
        data = torch.randn(64 * 8 * 32, dtype=torch.float16)
        block = pool.allocate_block(
            token_start=0,
            token_count=64,
            compressed_data=data,
        )
        
        usage = pool.get_memory_usage()
        assert usage["num_blocks"] == 1
        assert usage["used_bytes"] > 0

    def test_pool_clear_all(self):
        """Clear all blocks"""
        pool = CompressedKVPool(
            num_heads=8,
            head_dim=128,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
        )
        
        for i in range(3):
            data = torch.randn(64 * 8 * 32, dtype=torch.float16)
            pool.allocate_block(
                token_start=i * 64,
                token_count=64,
                compressed_data=data,
            )
        
        pool.clear_all()
        assert len(pool.blocks) == 0
        assert pool.stats.total_used_bytes == 0

    def test_pool_lazy_allocation(self):
        """Lazy allocation strategy"""
        pool = CompressedKVPool(
            num_heads=8,
            head_dim=128,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
            strategy=AllocationStrategy.LAZY,
        )
        
        # Pool tensor should not be pre-allocated
        assert pool.pool_tensor is None


class TestMultiLayerAllocator:
    """Test multi-layer KV allocator"""

    def test_multi_layer_init(self):
        """Initialize multi-layer allocator"""
        allocator = MultiLayerCompressedKVAllocator(
            num_layers=4,
            num_heads=8,
            head_dim=128,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
        )
        
        assert len(allocator.layer_allocators) == 4
        assert allocator.num_layers == 4

    def test_allocate_for_layer(self):
        """Allocate K and V for a specific layer"""
        allocator = MultiLayerCompressedKVAllocator(
            num_layers=4,
            num_heads=8,
            head_dim=128,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
        )
        
        k_data = torch.randn(64 * 8 * 32, dtype=torch.float16)
        v_data = torch.randn(64 * 8 * 32, dtype=torch.float16)
        
        k_block, v_block = allocator.allocate_for_layer(
            layer_id=0,
            token_start=0,
            token_count=64,
            compressed_k=k_data,
            compressed_v=v_data,
            compression_mode="tq2",
        )
        
        assert k_block is not None
        assert v_block is not None
        assert k_block.block_id != v_block.block_id

    def test_get_layer_allocator(self):
        """Get allocator for specific layer"""
        allocator = MultiLayerCompressedKVAllocator(
            num_layers=4,
            num_heads=8,
            head_dim=128,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
        )
        
        layer_0 = allocator.get_layer_allocator(0)
        layer_3 = allocator.get_layer_allocator(3)
        
        assert layer_0 is not None
        assert layer_3 is not None
        assert layer_0 is not layer_3

    def test_get_total_memory_usage(self):
        """Get aggregate memory usage across layers"""
        allocator = MultiLayerCompressedKVAllocator(
            num_layers=4,
            num_heads=8,
            head_dim=128,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
        )
        
        # Allocate for each layer
        for layer_id in range(4):
            k_data = torch.randn(64 * 8 * 32, dtype=torch.float16)
            v_data = torch.randn(64 * 8 * 32, dtype=torch.float16)
            allocator.allocate_for_layer(
                layer_id=layer_id,
                token_start=0,
                token_count=64,
                compressed_k=k_data,
                compressed_v=v_data,
            )
        
        total_usage = allocator.get_total_memory_usage()
        assert total_usage["total_blocks"] == 8  # 2 blocks per layer
        assert total_usage["layers"] == 4

    def test_release_all_layers(self):
        """Release all blocks across all layers"""
        allocator = MultiLayerCompressedKVAllocator(
            num_layers=4,
            num_heads=8,
            head_dim=128,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
        )
        
        # Allocate for each layer
        for layer_id in range(4):
            k_data = torch.randn(64 * 8 * 32, dtype=torch.float16)
            v_data = torch.randn(64 * 8 * 32, dtype=torch.float16)
            allocator.allocate_for_layer(
                layer_id=layer_id,
                token_start=0,
                token_count=64,
                compressed_k=k_data,
                compressed_v=v_data,
            )
        
        allocator.release_all()
        
        total_usage = allocator.get_total_memory_usage()
        assert total_usage["total_blocks"] == 0


class TestGlobalAllocator:
    """Test global allocator initialization"""

    def test_init_allocator(self):
        """Initialize global allocator"""
        allocator = init_compressed_kv_allocator(
            num_layers=4,
            num_heads=8,
            head_dim=128,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
        )
        
        assert allocator is not None
        assert get_compressed_kv_allocator() is allocator

    def test_allocator_invalid_layer(self):
        """Error on invalid layer ID"""
        allocator = init_compressed_kv_allocator(
            num_layers=4,
            num_heads=8,
            head_dim=128,
            max_total_tokens=1024,
            compression_ratio=0.25,
            device=torch.device("cpu"),
        )
        
        with pytest.raises(ValueError):
            allocator.get_layer_allocator(layer_id=10)


class TestAllocationStatistics:
    """Test statistics dataclass"""

    def test_utilization_percent(self):
        """Calculate utilization percentage"""
        stats = AllocationStatistics(
            total_allocated_bytes=1000,
            total_used_bytes=500,
        )
        
        assert stats.utilization_percent == 50.0

    def test_fragmentation_percent(self):
        """Calculate fragmentation percentage"""
        stats = AllocationStatistics(
            total_allocated_bytes=1000,
            total_used_bytes=500,
        )
        
        # Free bytes: 500, fragmentation: 50%
        assert stats.fragmentation_percent == 50.0

    def test_statistics_zero_allocated(self):
        """Handle zero allocation"""
        stats = AllocationStatistics(total_allocated_bytes=0, total_used_bytes=0)
        
        assert stats.utilization_percent == 0.0
        assert stats.fragmentation_percent == 0.0


class TestDtypeBytes:
    """Test dtype_bytes helper"""

    def test_float32(self):
        assert dtype_bytes(torch.float32) == 4

    def test_float16(self):
        assert dtype_bytes(torch.float16) == 2

    def test_int8(self):
        assert dtype_bytes(torch.int8) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
