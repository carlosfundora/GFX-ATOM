"""
SGLang KV Pool Allocator for Compressed Storage (Phase 4.4.3)

Manages memory allocation for compressed KV data with support for:
  - Paged allocation (matching SGLang's token-to-KV pool structure)
  - Compressed data storage (respecting compression ratios)
  - Metadata alignment (scale factors, zero points, etc.)
  - Garbage collection and eviction policies
  - Multi-layer support (one allocator per transformer layer)

Architecture:
  ReqToTokenPool → KV Block Allocator → Compressed KV Storage
                                              ↓
                                        TurboQuant Data
                                        + Scale/Zero metadata
                                        + Token indices

This module provides:
  - CompressedKVPool: Manages compressed tensors with paging
  - AllocationStrategy: Enum for allocation policies (eager, lazy, hybrid)
  - AllocationStatistics: Telemetry for allocation behavior
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """Memory allocation strategy for compressed KV"""
    EAGER = "eager"  # Pre-allocate entire capacity
    LAZY = "lazy"  # Allocate on-demand
    HYBRID = "hybrid"  # Start lazy, switch to eager if needed


@dataclass
class AllocationStatistics:
    """Telemetry for KV pool allocator"""
    total_allocated_bytes: int = 0
    total_used_bytes: int = 0
    num_allocations: int = 0
    num_deallocations: int = 0
    num_fragmentation_events: int = 0
    peak_usage_bytes: int = 0
    compression_ratio: float = 1.0  # bytes_stored / bytes_original
    
    @property
    def utilization_percent(self) -> float:
        """Percentage of allocated memory actually in use"""
        if self.total_allocated_bytes == 0:
            return 0.0
        return 100.0 * self.total_used_bytes / self.total_allocated_bytes
    
    @property
    def fragmentation_percent(self) -> float:
        """Percentage of allocated memory that's fragmented"""
        if self.total_allocated_bytes == 0:
            return 0.0
        free_bytes = self.total_allocated_bytes - self.total_used_bytes
        return 100.0 * free_bytes / self.total_allocated_bytes


@dataclass
class CompressedKVBlock:
    """Single block of compressed KV data"""
    block_id: int
    token_start: int  # Starting token index in sequence
    token_count: int  # Number of tokens in this block
    data: torch.Tensor  # Compressed data
    scale: Optional[torch.Tensor] = None  # Optional scale factors
    zero_point: Optional[torch.Tensor] = None  # Optional zero points
    compression_mode: str = "tq2"  # e.g., "tq2", "rq3_planar"
    
    def size_bytes(self) -> int:
        """Total size including metadata"""
        total = self.data.numel() * self.data.element_size()
        if self.scale is not None:
            total += self.scale.numel() * self.scale.element_size()
        if self.zero_point is not None:
            total += self.zero_point.numel() * self.zero_point.element_size()
        return total
    
    def is_full(self, page_size: int) -> bool:
        """Check if block has reached page size"""
        return self.token_count >= page_size


class CompressedKVPool:
    """
    Memory pool for compressed KV data with paging support.
    
    Similar to SGLang's ReqToTokenPool and token_to_kv_pool_allocator,
    but for compressed tensors.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,
        max_total_tokens: int = 32000,
        compression_ratio: float = 0.25,  # TQ2 typical
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
        strategy: AllocationStrategy = AllocationStrategy.LAZY,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_total_tokens = max_total_tokens
        self.compression_ratio = compression_ratio
        self.dtype = dtype
        self.device = device
        self.strategy = strategy
        
        # Calculate total capacity
        # Original KV: [num_tokens, num_heads, head_dim] in float32 (4 bytes)
        # Compressed: [num_tokens, num_heads, head_dim * compression_ratio]
        bytes_per_original_token = num_heads * head_dim * 4  # float32
        bytes_per_compressed_token = int(bytes_per_original_token * compression_ratio)
        self.bytes_per_token = bytes_per_compressed_token
        
        # Pool storage
        self.blocks: dict[int, CompressedKVBlock] = {}
        self.next_block_id = 0
        
        # Allocate backing storage
        total_bytes_needed = max_total_tokens * bytes_per_compressed_token
        num_elements = total_bytes_needed // dtype_bytes(dtype)
        
        if strategy == AllocationStrategy.EAGER:
            self.pool_tensor = torch.zeros(
                num_elements, dtype=dtype, device=device
            )
            logger.info(f"Eagerly allocated KV pool: {total_bytes_needed / 1e9:.2f} GB")
        else:
            self.pool_tensor = None
            self._allocated_elements = 0
            logger.info(
                f"Lazy KV pool allocator: will allocate up to {total_bytes_needed / 1e9:.2f} GB"
            )
        
        self.stats = AllocationStatistics(
            total_allocated_bytes=total_bytes_needed if strategy == AllocationStrategy.EAGER else 0,
            compression_ratio=compression_ratio,
        )
    
    def allocate_block(
        self,
        token_start: int,
        token_count: int,
        compressed_data: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None,
        compression_mode: str = "tq2",
    ) -> CompressedKVBlock:
        """
        Allocate a new block for compressed KV data.
        
        Args:
            token_start: Starting token index
            token_count: Number of tokens in this block
            compressed_data: Compressed tensor
            scale: Optional scale factors
            zero_point: Optional zero points
            compression_mode: Quantization mode (e.g., "tq2")
        
        Returns:
            CompressedKVBlock with allocated storage
        """
        block_id = self.next_block_id
        self.next_block_id += 1
        
        # Verify tensor size matches expected compression
        expected_elements = token_count * self.num_heads * int(self.head_dim * self.compression_ratio)
        actual_elements = compressed_data.numel()
        
        if actual_elements != expected_elements:
            logger.warning(
                f"Block {block_id}: expected {expected_elements} elements, "
                f"got {actual_elements}. Reshaping."
            )
            compressed_data = compressed_data.reshape(expected_elements)
        
        # Ensure data is on correct device and dtype
        compressed_data = compressed_data.to(device=self.device, dtype=self.dtype)
        if scale is not None:
            scale = scale.to(device=self.device)
        if zero_point is not None:
            zero_point = zero_point.to(device=self.device)
        
        # Create block
        block = CompressedKVBlock(
            block_id=block_id,
            token_start=token_start,
            token_count=token_count,
            data=compressed_data,
            scale=scale,
            zero_point=zero_point,
            compression_mode=compression_mode,
        )
        
        # Store block
        self.blocks[block_id] = block
        
        # Update statistics
        block_bytes = block.size_bytes()
        self.stats.total_used_bytes += block_bytes
        self.stats.num_allocations += 1
        if self.stats.total_used_bytes > self.stats.peak_usage_bytes:
            self.stats.peak_usage_bytes = self.stats.total_used_bytes
        
        logger.debug(f"Allocated KV block {block_id}: {token_count} tokens, {block_bytes:,} bytes")
        
        return block
    
    def get_block(self, block_id: int) -> Optional[CompressedKVBlock]:
        """Retrieve a block by ID"""
        return self.blocks.get(block_id)
    
    def release_block(self, block_id: int) -> bool:
        """
        Release a block and free its memory.
        
        Returns:
            True if block was released, False if not found
        """
        if block_id not in self.blocks:
            return False
        
        block = self.blocks.pop(block_id)
        block_bytes = block.size_bytes()
        self.stats.total_used_bytes -= block_bytes
        self.stats.num_deallocations += 1
        
        logger.debug(f"Released KV block {block_id}: freed {block_bytes:,} bytes")
        
        return True
    
    def clear_all(self):
        """Clear all blocks and reset pool"""
        self.blocks.clear()
        self.stats.total_used_bytes = 0
        self.stats.num_allocations = 0
        self.stats.num_deallocations = 0
        self.stats.num_fragmentation_events = 0
        logger.info("Cleared KV pool")
    
    def get_memory_usage(self) -> dict:
        """Return memory usage summary"""
        return {
            "allocated_bytes": self.stats.total_allocated_bytes,
            "used_bytes": self.stats.total_used_bytes,
            "free_bytes": self.stats.total_allocated_bytes - self.stats.total_used_bytes,
            "utilization_percent": self.stats.utilization_percent,
            "num_blocks": len(self.blocks),
            "peak_usage_bytes": self.stats.peak_usage_bytes,
        }
    
    def defragment(self) -> int:
        """
        Attempt to reduce fragmentation by compacting blocks.
        
        Returns:
            Number of blocks compacted
        """
        # STUB: Simple compaction not yet implemented
        # Phase 5 can add optimizations like block merging
        compacted = 0
        logger.debug(f"KV pool defragmentation: compacted {compacted} blocks")
        return compacted
    
    def get_stats(self) -> AllocationStatistics:
        """Return allocation statistics"""
        return self.stats


class MultiLayerCompressedKVAllocator:
    """
    Manages KV allocation across all transformer layers.
    
    Provides per-layer allocators and global coordination.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,
        max_total_tokens: int = 32000,
        compression_ratio: float = 0.25,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        self.num_layers = num_layers
        self.layer_allocators: List[CompressedKVPool] = []
        
        for layer_id in range(num_layers):
            allocator = CompressedKVPool(
                num_heads=num_heads,
                head_dim=head_dim,
                page_size=page_size,
                max_total_tokens=max_total_tokens,
                compression_ratio=compression_ratio,
                dtype=dtype,
                device=device,
            )
            self.layer_allocators.append(allocator)
    
    def allocate_for_layer(
        self,
        layer_id: int,
        token_start: int,
        token_count: int,
        compressed_k: torch.Tensor,
        compressed_v: torch.Tensor,
        scale_k: Optional[torch.Tensor] = None,
        scale_v: Optional[torch.Tensor] = None,
        compression_mode: str = "tq2",
    ) -> Tuple[CompressedKVBlock, CompressedKVBlock]:
        """Allocate both K and V blocks for a layer"""
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(f"Layer {layer_id} out of range [0, {self.num_layers})")
        
        allocator = self.layer_allocators[layer_id]
        
        k_block = allocator.allocate_block(
            token_start=token_start,
            token_count=token_count,
            compressed_data=compressed_k,
            scale=scale_k,
            zero_point=None,
            compression_mode=compression_mode,
        )
        
        v_block = allocator.allocate_block(
            token_start=token_start,
            token_count=token_count,
            compressed_data=compressed_v,
            scale=scale_v,
            zero_point=None,
            compression_mode=compression_mode,
        )
        
        return k_block, v_block
    
    def get_layer_allocator(self, layer_id: int) -> CompressedKVPool:
        """Get allocator for a specific layer"""
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(f"Layer {layer_id} out of range [0, {self.num_layers})")
        return self.layer_allocators[layer_id]
    
    def release_all(self):
        """Release all blocks across all layers"""
        for allocator in self.layer_allocators:
            allocator.clear_all()
    
    def get_total_memory_usage(self) -> dict:
        """Return aggregate memory usage across all layers"""
        total_allocated = 0
        total_used = 0
        total_blocks = 0
        
        for allocator in self.layer_allocators:
            usage = allocator.get_memory_usage()
            total_allocated += usage["allocated_bytes"]
            total_used += usage["used_bytes"]
            total_blocks += usage["num_blocks"]
        
        return {
            "total_allocated_bytes": total_allocated,
            "total_used_bytes": total_used,
            "total_free_bytes": total_allocated - total_used,
            "total_utilization_percent": (100.0 * total_used / total_allocated) if total_allocated > 0 else 0,
            "total_blocks": total_blocks,
            "layers": len(self.layer_allocators),
        }


def dtype_bytes(dtype: torch.dtype) -> int:
    """Return size in bytes for a torch dtype"""
    if dtype == torch.float32:
        return 4
    elif dtype == torch.float16:
        return 2
    elif dtype == torch.float8_e4m3fn:
        return 1
    elif dtype == torch.int8:
        return 1
    elif dtype == torch.int4:
        return 0.5  # Approximate
    else:
        return 4  # Default


# Global compressed KV allocator (Phase 4.5 feature gate)
_global_allocator: Optional[MultiLayerCompressedKVAllocator] = None


def init_compressed_kv_allocator(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    page_size: int = 16,
    max_total_tokens: int = 32000,
    compression_ratio: float = 0.25,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cuda"),
) -> MultiLayerCompressedKVAllocator:
    """Initialize global KV allocator"""
    global _global_allocator
    
    _global_allocator = MultiLayerCompressedKVAllocator(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_total_tokens=max_total_tokens,
        compression_ratio=compression_ratio,
        dtype=dtype,
        device=device,
    )
    
    logger.info(f"Initialized compressed KV allocator: {num_layers} layers, {max_total_tokens} max tokens")
    return _global_allocator


def get_compressed_kv_allocator() -> Optional[MultiLayerCompressedKVAllocator]:
    """Get global allocator (or None if not initialized)"""
    return _global_allocator
