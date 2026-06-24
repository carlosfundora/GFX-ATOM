"""
Integration Example: Using Tiered KV Cache with SGLang

This example demonstrates how the two-tier KV cache strategy
integrates with SGLang's backend adapter layer.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from tiered_kv_cache_manager import TieredKvCacheManager, CacheTier
from sglang_backend_adapter import TieredKvCacheAdapter


def example_long_context_inference():
    """
    Example: Processing a very long context with tiered caching.
    
    Scenario: 50K token context (hypothetical)
    - GPU tier (RotorQuant): holds recent/important prefixes
    - RAM tier (TurboQuant): holds older/less-critical sequences
    """
    print("\n" + "=" * 80)
    print("EXAMPLE: Long-Context Inference with Tiered KV Cache")
    print("=" * 80 + "\n")
    
    # Initialize tiered adapter (Tier 1: RotorQuant, Tier 2: TurboQuant)
    adapter = TieredKvCacheAdapter(
        gpu_capacity_mb=500,   # 500MB GPU (small for demo)
        ram_capacity_mb=2000,  # 2GB RAM
        primary_codec="rq3_planar",  # RotorQuant 3-bit
        secondary_codec="tq2",       # TurboQuant 2-bit fallback
        dimension=4096,
        num_heads=32,
    )
    
    print("✓ Tiered cache initialized")
    print(f"  Tier 1 (GPU): 500MB RotorQuant")
    print(f"  Tier 2 (RAM): 2000MB TurboQuant\n")
    
    # Simulate processing 50K tokens in 4K-token chunks
    chunk_size = 4096
    num_chunks = 12  # 12 × 4K = 48K tokens
    
    print(f"Processing {num_chunks} chunks of {chunk_size} tokens each ({num_chunks * chunk_size}K total):\n")
    
    block_ids = []
    for chunk_num in range(num_chunks):
        # Simulate KV tensors for this chunk
        k_cache = torch.randn(chunk_size, 4096)
        v_cache = torch.randn(chunk_size, 4096)
        
        # Allocate block (importance decreases for older chunks)
        importance = max(0.1, 1.0 - (chunk_num * 0.05))
        block_id = adapter.allocate_kv_block(
            request_id="long_context_req",
            layer_idx=0,
            k_cache=k_cache,
            v_cache=v_cache,
            importance_score=importance,
        )
        
        block_ids.append(block_id)
        
        # Get current stats
        stats = adapter.get_cache_stats()
        tier_1_blocks = stats["gpu_tier"]["blocks"]
        tier_2_blocks = stats["ram_tier"]["blocks"]
        tier_1_util = stats["gpu_tier"]["utilization_pct"]
        tier_2_util = stats["ram_tier"]["utilization_pct"]
        
        print(
            f"  Chunk {chunk_num:2d}: "
            f"GPU={tier_1_blocks:2d} blocks ({tier_1_util:5.1f}%), "
            f"RAM={tier_2_blocks:2d} blocks ({tier_2_util:5.1f}%)"
        )
    
    print("\n" + "-" * 80)
    print("FINAL CACHE STATE:")
    print("-" * 80 + "\n")
    
    adapter.print_cache_summary()
    
    print("\n" + "-" * 80)
    print("VERIFYING BLOCK ACCESS:")
    print("-" * 80 + "\n")
    
    # Access some blocks to demonstrate hit/miss behavior
    print("Accessing recent blocks (high hit rate expected):")
    for block_id in block_ids[-3:]:  # Last 3 blocks (most recent)
        try:
            result = adapter.get_kv_block(block_id)
            stats = adapter.get_cache_stats()
            print(f"  Block {block_id}: ✓ accessed (GPU hits: {stats['gpu_tier']['hits']})")
        except Exception as e:
            print(f"  Block {block_id}: ✗ error {e}")
    
    print("\nAccessing old blocks (potentially from RAM):")
    for block_id in block_ids[:2]:  # First 2 blocks (oldest)
        try:
            result = adapter.get_kv_block(block_id)
            stats = adapter.get_cache_stats()
            print(f"  Block {block_id}: ✓ accessed (RAM misses: {stats['ram_tier']['misses']})")
        except Exception as e:
            print(f"  Block {block_id}: ✗ error {e}")
    
    print("\n" + "=" * 80)
    print("✓ Integration example completed successfully")
    print("=" * 80 + "\n")


def example_importance_weighted_selection():
    """
    Example: Using importance-weighted attention to guide tier placement.
    
    Scenario: Model identifies which tokens are critical for attention.
    Critical tokens stay in GPU tier; less critical go to RAM.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE: Importance-Weighted Tier Selection")
    print("=" * 80 + "\n")
    
    cache_mgr = TieredKvCacheManager(
        gpu_tier_capacity_mb=200,
        ram_tier_capacity_mb=500,
    )
    
    print("✓ Cache manager initialized (GPU=200MB, RAM=500MB)\n")
    
    # Simulate blocks with varying importance (e.g., from attention head analysis)
    block_configs = [
        ("critical_prefix", 0.95),    # Very important
        ("normal_context", 0.50),     # Normal importance
        ("background_noise", 0.05),   # Low importance
    ]
    
    print("Allocating blocks with importance weights:\n")
    
    block_ids = {}
    for name, importance in block_configs:
        data = torch.randn(16, 2048)
        block_id = cache_mgr.allocate_block(
            request_id=f"req_{name}",
            layer_idx=0,
            seq_start=0,
            seq_end=16,
            data=data,
            importance_score=importance,
        )
        block_ids[name] = block_id
        tier = cache_mgr.block_to_tier[block_id]
        print(f"  {name:20s}: importance={importance:.2f} → {tier.value}")
    
    print("\n" + "-" * 80)
    print("EVICTION TEST (showing importance-weighted behavior):")
    print("-" * 80 + "\n")
    
    # Try to force eviction
    try:
        cache_mgr._evict_to_make_space(1024 * 1024, CacheTier.GPU_ROTOR)
        
        print("After eviction attempt:")
        for name, block_id in block_ids.items():
            if cache_mgr.block_to_tier.get(block_id) is not None:
                tier = cache_mgr.block_to_tier[block_id]
                print(f"  {name:20s}: still in {tier.value} tier")
            else:
                print(f"  {name:20s}: evicted")
    except Exception as e:
        print(f"  Eviction not triggered (cache has space): {e}")
    
    print("\n" + "=" * 80)
    print("✓ Importance-weighted example completed")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("\n╔" + "=" * 78 + "╗")
    print("║" + " TIERED KV CACHE INTEGRATION EXAMPLES ".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    
    example_long_context_inference()
    example_importance_weighted_selection()
    
    print("\n✅ All integration examples completed successfully!\n")
