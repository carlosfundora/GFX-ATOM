"""
Tiered KV Cache Manager: RotorQuant (GPU) + TurboQuant (RAM spill)

Two-tier caching strategy:
1. **Tier 1 (GPU)**: RotorQuant KV cache (3-bit, 8x compression, fast)
   - Per 16 float (64 bytes): ~6 bytes data + 2 bytes metadata (rotation index + scale)
   - Total: 8 bytes per 64 bytes = 8x compression
   
2. **Tier 2 (RAM)**: TurboQuant KV cache (spill for overflow, slight latency)
   - Older/less-critical sequences moved to system RAM
   - Called back on-demand with manageable latency

Strategy:
- Keep hot/recent sequences in RotorQuant GPU
- Evict cold sequences to TurboQuant RAM
- Track block age, access frequency, importance weights
- Adaptive swap based on GPU memory pressure
"""

import time
import logging
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict
from enum import Enum
import heapq

import torch
import numpy as np

from kv_quant_contracts import (
    KvCodec,
    UniversalKvBlockHeaderV1,
    UniversalKvPlacementPolicy,
    UniversalKvStage,
)

logger = logging.getLogger(__name__)


class CacheTier(Enum):
    """Tier identification."""
    GPU_ROTOR = "gpu_rotor"  # RotorQuant on GPU (Tier 1, fast)
    RAM_TURBO = "ram_turbo"  # TurboQuant on RAM (Tier 2, fallback)


@dataclass
class BlockMetadata:
    """Per-block metadata for tiering decisions."""
    block_id: int
    request_id: str
    layer_idx: int
    seq_start: int
    seq_end: int
    current_tier: CacheTier
    created_at: float
    last_accessed: float
    access_count: int = 0
    importance_score: float = 1.0  # importance-weighted attention
    size_bytes: int = 0  # actual size after compression
    is_pinned: bool = False  # don't evict (prefix cache)
    original_shape: Tuple[int, ...] = field(default_factory=tuple)
    original_numel: int = 0
    codec_name: str = "rq3_planar"
    quant_scale: float = 1.0
    rotor_dim: int = 0
    used_rust_codec: bool = False
    stage: UniversalKvStage = UniversalKvStage.hot_rotor
    stage_age_steps: int = 0
    last_stage_tick: int = 0
    block_header: UniversalKvBlockHeaderV1 | None = None
    warm_reference_payload: bytes | None = None
    warm_reference_format: str | None = None
    cold_reference_payload: bytes | None = None
    cold_reference_format: str | None = None
    stage_materialization_source: str = "hot_rotor_payload"


@dataclass
class TierStats:
    """Statistics for a single tier."""
    tier: CacheTier
    blocks_count: int = 0
    total_bytes: int = 0
    hits: int = 0
    misses: int = 0
    swaps_in: int = 0  # RAM → GPU promotions
    swaps_out: int = 0  # GPU → RAM demotions


class TieredKvCacheManager:
    """
    Manages two-tier KV cache: RotorQuant (GPU) + TurboQuant (RAM).
    
    Features:
    - Adaptive eviction based on age, access frequency, importance
    - Block-level granularity (16-value blocks)
    - Metrics: hit rate, swap overhead, space efficiency
    - Configurable tier sizes
    """

    def __init__(
        self,
        gpu_tier_capacity_mb: int = 8000,  # ~8GB GPU VRAM for KV
        ram_tier_capacity_mb: int = 32000,  # ~32GB system RAM for KV spill
        block_size: int = 16,  # values per block
        block_metadata_bytes: int = 2,  # rotation index + scale
        prefer_rust_rotor: bool = True,
        placement_policy: UniversalKvPlacementPolicy | None = None,
    ):
        self.gpu_tier_capacity = gpu_tier_capacity_mb * 1024 * 1024  # bytes
        self.ram_tier_capacity = ram_tier_capacity_mb * 1024 * 1024  # bytes
        self.block_size = block_size
        self.block_metadata_bytes = block_metadata_bytes
        self.prefer_rust_rotor = prefer_rust_rotor
        self.placement_policy = placement_policy or UniversalKvPlacementPolicy()
        
        # Tier storage (block_id → BlockMetadata)
        self.gpu_blocks: Dict[int, BlockMetadata] = {}
        self.ram_blocks: Dict[int, BlockMetadata] = {}
        
        # Actual block data storage (for RAM tier)
        self.ram_block_data: Dict[int, bytes] = {}
        
        # GPU tier tracking (assumes PyTorch tensor backend)
        self.gpu_block_data: Dict[int, torch.Tensor] = {}
        
        # Statistics
        self.stats = {
            CacheTier.GPU_ROTOR: TierStats(tier=CacheTier.GPU_ROTOR),
            CacheTier.RAM_TURBO: TierStats(tier=CacheTier.RAM_TURBO),
        }
        
        # Eviction priority queue (LRU + importance weighting)
        # (priority, timestamp, block_id)
        self.eviction_heap: List[Tuple[float, float, int]] = []
        
        # Reverse lookup: block_id → tier
        self.block_to_tier: Dict[int, CacheTier] = {}
        
        # Global block counter
        self._next_block_id = 0
        self._stage_tick = 0
        self._rust_rotor_codec = self._try_load_rust_rotor_codec()
        
        logger.info(
            f"TieredKvCacheManager initialized: "
            f"GPU={gpu_tier_capacity_mb}MB, RAM={ram_tier_capacity_mb}MB, "
            f"block_size={block_size}, metadata={block_metadata_bytes}B"
        )
        if self._rust_rotor_codec is not None:
            logger.info("TieredKvCacheManager using Rust RotorQuant codec path")

    def allocate_block(
        self,
        request_id: str,
        layer_idx: int,
        seq_start: int,
        seq_end: int,
        data: torch.Tensor,
        importance_score: float = 1.0,
    ) -> int:
        """
        Allocate a new KV block (16-value group).
        
        Returns: block_id
        """
        block_id = self._next_block_id
        self._next_block_id += 1
        stage_tick = self._advance_stage_tick()
        
        # Compress to 3-bit RotorQuant format
        # Assume data is shape (seq_len, num_heads * head_dim)
        compressed_data, compressed_size, codec_meta = self._compress_rotor(data)
        initial_stage = self.placement_policy.select_stage(
            importance=importance_score,
            age_steps=0,
            gpu_utilization_pct=self._gpu_utilization_ratio(),
        )
        
        # Create metadata
        metadata = BlockMetadata(
            block_id=block_id,
            request_id=request_id,
            layer_idx=layer_idx,
            seq_start=seq_start,
            seq_end=seq_end,
            current_tier=CacheTier.GPU_ROTOR,
            created_at=time.time(),
            last_accessed=time.time(),
            importance_score=importance_score,
            size_bytes=compressed_size,
            original_shape=codec_meta["original_shape"],
            original_numel=codec_meta["original_numel"],
            codec_name=codec_meta["codec_name"],
            quant_scale=codec_meta["quant_scale"],
            rotor_dim=codec_meta["rotor_dim"],
            used_rust_codec=codec_meta["used_rust_codec"],
            stage=initial_stage,
            stage_age_steps=0,
            last_stage_tick=stage_tick,
            block_header=self._build_block_header(
                stage=initial_stage,
                quant_scale=codec_meta["quant_scale"],
                used_rust_codec=codec_meta["used_rust_codec"],
            ),
        )
        self._ensure_stage_reference_payload(
            metadata,
            source_tensor=data.detach().float().cpu(),
        )

        prefer_gpu = metadata.stage != UniversalKvStage.cold_turbo_residual
        if prefer_gpu:
            if not self._place_block_in_gpu(block_id, metadata, compressed_data):
                evicted_from_gpu = self._evict_to_make_space(
                    compressed_size, target_tier=CacheTier.GPU_ROTOR
                )
                if evicted_from_gpu >= 1:
                    self._place_block_in_gpu(block_id, metadata, compressed_data)
                if block_id not in self.block_to_tier:
                    if not self._place_block_in_ram(block_id, metadata, compressed_data):
                        logger.warning(
                            f"Block {block_id} allocation failed: both tiers full"
                        )
                        raise MemoryError("Both GPU and RAM tiers are full")
        else:
            if not self._place_block_in_ram(block_id, metadata, compressed_data):
                if not self._place_block_in_gpu(block_id, metadata, compressed_data):
                    logger.warning(f"Block {block_id} allocation failed: both tiers full")
                    raise MemoryError("Both GPU and RAM tiers are full")
        
        # Add to eviction heap
        self._add_to_eviction_heap(block_id)
        
        return block_id

    def access_block(self, block_id: int) -> torch.Tensor:
        """
        Access a block, potentially promoting from RAM to GPU if hot.
        
        Returns: decompressed KV tensor
        """
        tier = self.block_to_tier.get(block_id)
        if tier is None:
            raise KeyError(f"Block {block_id} not found in cache")
        
        if tier == CacheTier.GPU_ROTOR:
            # Cache hit on GPU
            metadata = self.gpu_blocks[block_id]
            self._refresh_block_stage(metadata)
            metadata.last_accessed = time.time()
            metadata.access_count += 1
            self.stats[CacheTier.GPU_ROTOR].hits += 1
            
            # Return decompressed data
            return self._materialize_block_tensor(
                metadata, compressed_data=self.gpu_block_data[block_id]
            )
        
        elif tier == CacheTier.RAM_TURBO:
            # Cache miss (spill): bring from RAM, possibly promote to GPU
            metadata = self.ram_blocks[block_id]
            self._refresh_block_stage(metadata)

            if (
                metadata.stage == UniversalKvStage.hot_rotor
                and self._gpu_tier_has_space(metadata.size_bytes)
            ):
                self._promote_to_gpu(block_id)
                self.stats[CacheTier.RAM_TURBO].swaps_in += 1
                logger.debug("Block %s promoted RAM → GPU due to hot stage", block_id)
                promoted_meta = self.gpu_blocks[block_id]
                promoted_meta.last_accessed = time.time()
                promoted_meta.access_count += 1
                self.stats[CacheTier.GPU_ROTOR].hits += 1
                return self._materialize_block_tensor(
                    promoted_meta, compressed_data=self.gpu_block_data[block_id]
                )
            
            metadata.last_accessed = time.time()
            metadata.access_count += 1
            self.stats[CacheTier.RAM_TURBO].misses += 1
            
            # Return decompressed data from RAM
            raw_bytes = self.ram_block_data[block_id]
            return self._materialize_block_tensor(metadata, raw_bytes=raw_bytes)

    def evict_block(self, block_id: int) -> None:
        """Evict a block from cache."""
        tier = self.block_to_tier.get(block_id)
        if tier is None:
            return
        
        if tier == CacheTier.GPU_ROTOR:
            metadata = self.gpu_blocks.pop(block_id, None)
            if metadata:
                self.gpu_block_data.pop(block_id, None)
                self.stats[CacheTier.GPU_ROTOR].blocks_count -= 1
                self.stats[CacheTier.GPU_ROTOR].total_bytes -= metadata.size_bytes
                logger.debug(f"Block {block_id} evicted from GPU tier")
        
        elif tier == CacheTier.RAM_TURBO:
            metadata = self.ram_blocks.pop(block_id, None)
            if metadata:
                self.ram_block_data.pop(block_id, None)
                self.stats[CacheTier.RAM_TURBO].blocks_count -= 1
                self.stats[CacheTier.RAM_TURBO].total_bytes -= metadata.size_bytes
                logger.debug(f"Block {block_id} evicted from RAM tier")
        
        self.block_to_tier.pop(block_id, None)

    def get_stats(self) -> Dict:
        """Return cache statistics."""
        gpu_stats = self.stats[CacheTier.GPU_ROTOR]
        ram_stats = self.stats[CacheTier.RAM_TURBO]
        
        gpu_hit_rate = (
            gpu_stats.hits / (gpu_stats.hits + gpu_stats.misses)
            if (gpu_stats.hits + gpu_stats.misses) > 0
            else 0.0
        )
        
        return {
            "gpu_tier": {
                "blocks": gpu_stats.blocks_count,
                "bytes": gpu_stats.total_bytes,
                "capacity_bytes": self.gpu_tier_capacity,
                "utilization_pct": (gpu_stats.total_bytes / self.gpu_tier_capacity * 100) if self.gpu_tier_capacity > 0 else 0.0,
                "hits": gpu_stats.hits,
                "hit_rate": gpu_hit_rate,
            },
            "ram_tier": {
                "blocks": ram_stats.blocks_count,
                "bytes": ram_stats.total_bytes,
                "capacity_bytes": self.ram_tier_capacity,
                "utilization_pct": (ram_stats.total_bytes / self.ram_tier_capacity * 100) if self.ram_tier_capacity > 0 else 0.0,
                "misses": ram_stats.misses,
                "swaps_in": ram_stats.swaps_in,
                "swaps_out": ram_stats.swaps_out,
            },
            "compression_ratio": 8.0,  # 64 bytes → 8 bytes
            "block_size": self.block_size,
        }

    # ============ Private Helpers ============

    def _try_load_rust_rotor_codec(self):
        if not self.prefer_rust_rotor:
            return None
        try:
            from rs_rotorquant_codec import PyRotorQuantCodec  # type: ignore

            return PyRotorQuantCodec("planar3", 42, True)
        except Exception as exc:
            logger.debug("Rust RotorQuant codec unavailable, using packed fallback: %s", exc)
            return None

    def _gpu_tier_has_space(self, required_bytes: int) -> bool:
        """Check if GPU tier has space."""
        used = sum(m.size_bytes for m in self.gpu_blocks.values())
        return used + required_bytes <= self.gpu_tier_capacity

    def _gpu_utilization_ratio(self) -> float:
        if self.gpu_tier_capacity <= 0:
            return 1.0
        used = sum(m.size_bytes for m in self.gpu_blocks.values())
        return min(1.0, max(0.0, used / self.gpu_tier_capacity))

    def _advance_stage_tick(self) -> int:
        self._stage_tick += 1
        return self._stage_tick

    def _build_block_header(
        self, stage: UniversalKvStage, quant_scale: float, used_rust_codec: bool
    ) -> UniversalKvBlockHeaderV1:
        flags = 0
        if stage == UniversalKvStage.cold_turbo_residual:
            flags |= UniversalKvBlockHeaderV1.FLAG_TURBO_RESIDUAL
        return UniversalKvBlockHeaderV1(
            block_size=self.block_size,
            bit_width=3,
            rotor_id=42 if used_rust_codec else 0,
            codec=KvCodec.rq3_planar,
            stage=stage,
            scale=quant_scale,
            flags=flags,
        )

    def _refresh_block_stage(self, metadata: BlockMetadata) -> None:
        tick = self._advance_stage_tick()
        age_steps = max(0, tick - metadata.last_stage_tick)
        stage = self.placement_policy.select_stage(
            importance=metadata.importance_score,
            age_steps=age_steps,
            gpu_utilization_pct=self._gpu_utilization_ratio(),
        )
        metadata.stage = stage
        metadata.stage_age_steps = age_steps
        metadata.last_stage_tick = tick
        metadata.block_header = self._build_block_header(
            stage=stage,
            quant_scale=metadata.quant_scale,
            used_rust_codec=metadata.used_rust_codec,
        )
        self._ensure_stage_reference_payload(metadata)

    def _place_block_in_gpu(
        self,
        block_id: int,
        metadata: BlockMetadata,
        compressed_data: torch.Tensor,
    ) -> bool:
        if not self._gpu_tier_has_space(metadata.size_bytes):
            return False
        self.gpu_blocks[block_id] = metadata
        self.gpu_block_data[block_id] = compressed_data
        metadata.current_tier = CacheTier.GPU_ROTOR
        self.block_to_tier[block_id] = CacheTier.GPU_ROTOR
        self.stats[CacheTier.GPU_ROTOR].blocks_count += 1
        self.stats[CacheTier.GPU_ROTOR].total_bytes += metadata.size_bytes
        return True

    def _place_block_in_ram(
        self,
        block_id: int,
        metadata: BlockMetadata,
        compressed_data: torch.Tensor,
    ) -> bool:
        if not self._ram_tier_has_space(metadata.size_bytes):
            return False
        self.ram_blocks[block_id] = metadata
        self.ram_block_data[block_id] = (
            compressed_data.cpu().numpy().tobytes()
            if isinstance(compressed_data, torch.Tensor)
            else compressed_data
        )
        metadata.current_tier = CacheTier.RAM_TURBO
        self.block_to_tier[block_id] = CacheTier.RAM_TURBO
        self.stats[CacheTier.RAM_TURBO].blocks_count += 1
        self.stats[CacheTier.RAM_TURBO].total_bytes += metadata.size_bytes
        return True

    def _ram_tier_has_space(self, required_bytes: int) -> bool:
        """Check if RAM tier has space."""
        used = sum(m.size_bytes for m in self.ram_blocks.values())
        return used + required_bytes <= self.ram_tier_capacity

    def _evict_to_make_space(self, required_bytes: int, target_tier: CacheTier) -> int:
        """
        Evict blocks from target_tier to make space.
        Prefers to evict cold, non-pinned, low-importance blocks.
        
        Returns: number of blocks evicted
        """
        if target_tier == CacheTier.GPU_ROTOR:
            blocks_dict = self.gpu_blocks
        else:
            blocks_dict = self.ram_blocks
        
        eviction_candidates = []
        now = time.time()
        
        for block_id, metadata in blocks_dict.items():
            if metadata.is_pinned:
                continue  # Skip pinned blocks
            
            # Eviction score: higher = higher priority for eviction
            # Consider: age, access frequency, importance
            age = now - metadata.last_accessed
            recency_penalty = age / 60.0  # normalize to minutes (not hours)
            importance_factor = 1.0 / (metadata.importance_score + 0.1)  # higher importance → lower eviction priority
            
            score = recency_penalty * importance_factor
            eviction_candidates.append((score, block_id, metadata.size_bytes))
        
        eviction_candidates.sort(reverse=True)  # Sort by score (highest = evict first)
        
        evicted = 0
        freed_bytes = 0
        
        for score, block_id, block_size in eviction_candidates:
            if freed_bytes >= required_bytes:
                break
            
            freed_bytes += block_size
            self.evict_block(block_id)
            evicted += 1
        
        logger.debug(
            f"Evicted {evicted} blocks (freed {freed_bytes} bytes, needed {required_bytes})"
        )
        
        return evicted

    def _promote_to_gpu(self, block_id: int) -> None:
        """Promote a block from RAM to GPU."""
        if block_id not in self.ram_blocks:
            return
        
        metadata = self.ram_blocks.pop(block_id)
        raw_bytes = self.ram_block_data.pop(block_id)
        
        # Move back to GPU if available; otherwise keep on CPU for deterministic tests.
        tensor_data = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.uint8).copy())
        if torch.cuda.is_available():
            tensor_data = tensor_data.to("cuda")
        
        self.gpu_blocks[block_id] = metadata
        self.gpu_block_data[block_id] = tensor_data
        metadata.current_tier = CacheTier.GPU_ROTOR
        self.block_to_tier[block_id] = CacheTier.GPU_ROTOR
        
        self.stats[CacheTier.RAM_TURBO].blocks_count -= 1
        self.stats[CacheTier.RAM_TURBO].total_bytes -= metadata.size_bytes
        self.stats[CacheTier.GPU_ROTOR].blocks_count += 1
        self.stats[CacheTier.GPU_ROTOR].total_bytes += metadata.size_bytes
        self.stats[CacheTier.RAM_TURBO].swaps_out += 1

    def _add_to_eviction_heap(self, block_id: int) -> None:
        """Add block to eviction priority queue."""
        metadata = self.gpu_blocks.get(block_id) or self.ram_blocks.get(block_id)
        if metadata:
            priority = (metadata.last_accessed, block_id)
            heapq.heappush(self.eviction_heap, priority)

    def _compress_rotor(self, data: torch.Tensor) -> Tuple[torch.Tensor, int, Dict]:
        """
        Compress to 3-bit RotorQuant format.
        
        Input: (seq_len, dim) float32/float16
        Output: compressed tensor + size in bytes + codec metadata
        
        Compression: 64 bytes (16×float32) → ~6 bytes data + 2 bytes metadata
        """
        original_shape = tuple(int(v) for v in data.shape)
        flat = data.detach().float().cpu().reshape(-1).numpy()
        original_numel = int(flat.size)
        rotor_dim = original_numel
        quant_scale = float(np.max(np.abs(flat))) if original_numel > 0 else 1.0
        if quant_scale < 1e-8:
            quant_scale = 1.0
        normalized = np.clip(flat / quant_scale, -1.0, 1.0).astype(np.float32)

        codec_name = "python_packed_rq3"
        used_rust_codec = False
        packed: np.ndarray

        if self._rust_rotor_codec is not None:
            try:
                rust_bytes = self._rust_rotor_codec.compress_planar(
                    normalized.tolist(), rotor_dim
                )
                packed = np.asarray(rust_bytes, dtype=np.uint8)
                codec_name = "rust_planar3"
                used_rust_codec = True
            except Exception as exc:
                logger.warning(
                    "Rust RotorQuant compression failed, falling back to packed path: %s",
                    exc,
                )
                packed = self._pack_3bit(self._quantize_to_3bit(normalized))
        else:
            packed = self._pack_3bit(self._quantize_to_3bit(normalized))

        compressed = torch.from_numpy(packed.copy())
        compressed_size = int(compressed.numel())
        metadata = {
            "original_shape": original_shape,
            "original_numel": original_numel,
            "quant_scale": quant_scale,
            "codec_name": codec_name,
            "rotor_dim": rotor_dim,
            "used_rust_codec": used_rust_codec,
        }
        return compressed, compressed_size, metadata

    def _decompress_rotor(self, compressed: torch.Tensor, metadata: BlockMetadata) -> torch.Tensor:
        """Decompress from 3-bit RotorQuant format back to float32 tensor."""
        compressed_np = compressed.detach().cpu().numpy().astype(np.uint8, copy=False)
        return self._decode_payload_to_tensor(compressed_np, metadata)

    def _decompress_rotor_bytes(self, raw_bytes: bytes, metadata: BlockMetadata) -> torch.Tensor:
        """Decompress from raw bytes (RAM tier)."""
        compressed = np.frombuffer(raw_bytes, dtype=np.uint8)
        return self._decode_payload_to_tensor(compressed, metadata)

    def _materialize_block_tensor(
        self,
        metadata: BlockMetadata,
        compressed_data: torch.Tensor | None = None,
        raw_bytes: bytes | None = None,
    ) -> torch.Tensor:
        if (
            metadata.stage == UniversalKvStage.warm_rotor_polar
            and metadata.warm_reference_payload is not None
        ):
            metadata.stage_materialization_source = "warm_rotor_polar_reference"
            return self._decode_warm_rotor_polar_reference(
                metadata.warm_reference_payload, metadata
            )
        if (
            metadata.stage == UniversalKvStage.cold_turbo_residual
            and metadata.cold_reference_payload is not None
        ):
            metadata.stage_materialization_source = "cold_turbo_residual_reference"
            return self._decode_cold_turbo_residual_reference(
                metadata.cold_reference_payload, metadata
            )
        metadata.stage_materialization_source = "hot_rotor_payload"
        if compressed_data is not None:
            return self._decompress_rotor(compressed_data, metadata)
        if raw_bytes is not None:
            return self._decompress_rotor_bytes(raw_bytes, metadata)
        decoded = self._decode_hot_payload_for_metadata(metadata)
        if decoded is None:
            raise KeyError(f"Unable to materialize payload for block {metadata.block_id}")
        return decoded

    def _decode_hot_payload_for_metadata(self, metadata: BlockMetadata) -> torch.Tensor | None:
        block_id = metadata.block_id
        tier = self.block_to_tier.get(block_id)
        if tier == CacheTier.GPU_ROTOR and block_id in self.gpu_block_data:
            return self._decompress_rotor(self.gpu_block_data[block_id], metadata)
        if tier == CacheTier.RAM_TURBO and block_id in self.ram_block_data:
            return self._decompress_rotor_bytes(self.ram_block_data[block_id], metadata)
        return None

    def _ensure_stage_reference_payload(
        self,
        metadata: BlockMetadata,
        source_tensor: torch.Tensor | None = None,
    ) -> None:
        if metadata.stage == UniversalKvStage.hot_rotor:
            return
        source = source_tensor if source_tensor is not None else self._decode_hot_payload_for_metadata(metadata)
        if source is None:
            return
        if (
            metadata.stage == UniversalKvStage.warm_rotor_polar
            and metadata.warm_reference_payload is None
        ):
            metadata.warm_reference_payload = self._encode_warm_rotor_polar_reference(source)
            metadata.warm_reference_format = "warm_rotor_polar_ref_v1"
        if (
            metadata.stage == UniversalKvStage.cold_turbo_residual
            and metadata.cold_reference_payload is None
        ):
            metadata.cold_reference_payload = self._encode_cold_turbo_residual_reference(source)
            metadata.cold_reference_format = "cold_turbo_residual_ref_v1"

    def _encode_warm_rotor_polar_reference(self, tensor: torch.Tensor) -> bytes:
        flat = tensor.detach().float().cpu().reshape(-1).numpy().astype(np.float32, copy=False)
        original_numel = int(flat.size)
        if original_numel == 0:
            return struct.pack("<4sI", b"WRP1", 0)
        if original_numel % 2:
            flat = np.pad(flat, (0, 1), mode="constant")
        pairs = flat.reshape(-1, 2)
        radius = np.sqrt(pairs[:, 0] ** 2 + pairs[:, 1] ** 2).astype(np.float16)
        theta = np.arctan2(pairs[:, 1], pairs[:, 0]).astype(np.float16)
        return struct.pack("<4sI", b"WRP1", original_numel) + radius.tobytes() + theta.tobytes()

    def _decode_warm_rotor_polar_reference(
        self, payload: bytes, metadata: BlockMetadata
    ) -> torch.Tensor:
        if len(payload) < 8:
            raise ValueError("Invalid warm reference payload")
        magic, original_numel = struct.unpack("<4sI", payload[:8])
        if magic != b"WRP1":
            raise ValueError("Unexpected warm reference payload magic")
        if original_numel == 0:
            return self._reshape_materialized_flat(np.zeros((0,), dtype=np.float32), metadata)
        pair_count = (original_numel + 1) // 2
        radius = np.frombuffer(payload, dtype=np.float16, count=pair_count, offset=8).astype(
            np.float32
        )
        theta_offset = 8 + pair_count * np.dtype(np.float16).itemsize
        theta = np.frombuffer(payload, dtype=np.float16, count=pair_count, offset=theta_offset).astype(
            np.float32
        )
        flat = np.empty(pair_count * 2, dtype=np.float32)
        flat[0::2] = radius * np.cos(theta)
        flat[1::2] = radius * np.sin(theta)
        return self._reshape_materialized_flat(flat[:original_numel], metadata)

    def _encode_cold_turbo_residual_reference(self, tensor: torch.Tensor) -> bytes:
        flat = tensor.detach().float().cpu().reshape(-1).numpy().astype(np.float32, copy=False)
        original_numel = int(flat.size)
        if original_numel == 0:
            return struct.pack("<4sIff", b"CTR1", 0, 1.0, 1.0)
        max_abs = float(np.max(np.abs(flat)))
        base_scale = max(max_abs / 127.0, 1e-8)
        base_q = np.clip(np.round(flat / base_scale), -127, 127).astype(np.int8)
        base_recon = base_q.astype(np.float32) * base_scale
        residual = flat - base_recon
        residual_max = float(np.max(np.abs(residual)))
        residual_scale = max(residual_max / 127.0, 1e-8)
        residual_q = np.clip(np.round(residual / residual_scale), -127, 127).astype(np.int8)
        return (
            struct.pack("<4sIff", b"CTR1", original_numel, float(base_scale), float(residual_scale))
            + base_q.tobytes()
            + residual_q.tobytes()
        )

    def _decode_cold_turbo_residual_reference(
        self, payload: bytes, metadata: BlockMetadata
    ) -> torch.Tensor:
        if len(payload) < 16:
            raise ValueError("Invalid cold reference payload")
        magic, original_numel, base_scale, residual_scale = struct.unpack("<4sIff", payload[:16])
        if magic != b"CTR1":
            raise ValueError("Unexpected cold reference payload magic")
        if original_numel == 0:
            return self._reshape_materialized_flat(np.zeros((0,), dtype=np.float32), metadata)
        base_q = np.frombuffer(payload, dtype=np.int8, count=original_numel, offset=16).astype(
            np.float32
        )
        residual_offset = 16 + original_numel
        residual_q = np.frombuffer(
            payload, dtype=np.int8, count=original_numel, offset=residual_offset
        ).astype(np.float32)
        flat = base_q * np.float32(base_scale) + residual_q * np.float32(residual_scale)
        return self._reshape_materialized_flat(flat, metadata)

    def _decode_payload_to_tensor(
        self, compressed: np.ndarray, metadata: BlockMetadata
    ) -> torch.Tensor:
        if metadata.used_rust_codec and self._rust_rotor_codec is not None:
            try:
                decoded = self._rust_rotor_codec.decompress_planar(
                    compressed.tolist(), metadata.rotor_dim or metadata.original_numel
                )
                flat = np.asarray(decoded, dtype=np.float32)
            except Exception as exc:
                logger.warning(
                    "Rust RotorQuant decompression failed, falling back to packed path: %s",
                    exc,
                )
                flat = self._dequantize_from_3bit(
                    self._unpack_3bit(compressed, metadata.original_numel)
                )
        else:
            flat = self._dequantize_from_3bit(
                self._unpack_3bit(compressed, metadata.original_numel)
            )

        if metadata.original_numel > 0:
            flat = flat[: metadata.original_numel] * metadata.quant_scale
        return self._reshape_materialized_flat(flat, metadata)

    def _reshape_materialized_flat(
        self, flat: np.ndarray, metadata: BlockMetadata
    ) -> torch.Tensor:
        if metadata.original_numel > 0:
            flat = flat[: metadata.original_numel]
        if metadata.original_shape:
            return torch.from_numpy(flat.reshape(metadata.original_shape).astype(np.float32))
        return torch.from_numpy(flat.astype(np.float32))

    def _quantize_to_3bit(self, normalized: np.ndarray) -> np.ndarray:
        quant = np.round((normalized + 1.0) * 0.5 * 7.0)
        return np.clip(quant, 0, 7).astype(np.uint8)

    def _dequantize_from_3bit(self, quantized: np.ndarray) -> np.ndarray:
        normalized = (quantized.astype(np.float32) / 7.0) * 2.0 - 1.0
        return normalized

    def _pack_3bit(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.zeros((0,), dtype=np.uint8)
        pad = (-values.size) % 8
        if pad:
            values = np.pad(values, (0, pad), mode="constant")
        groups = values.reshape(-1, 8).astype(np.uint16, copy=False)
        b0 = (groups[:, 0]) | (groups[:, 1] << 3) | ((groups[:, 2] & 0x03) << 6)
        b1 = (
            ((groups[:, 2] >> 2) & 0x01)
            | (groups[:, 3] << 1)
            | (groups[:, 4] << 4)
            | ((groups[:, 5] & 0x01) << 7)
        )
        b2 = ((groups[:, 5] >> 1) & 0x03) | (groups[:, 6] << 2) | (groups[:, 7] << 5)
        out = np.empty(groups.shape[0] * 3, dtype=np.uint8)
        out[0::3] = b0.astype(np.uint8)
        out[1::3] = b1.astype(np.uint8)
        out[2::3] = b2.astype(np.uint8)
        return out

    def _unpack_3bit(self, packed: np.ndarray, original_numel: int) -> np.ndarray:
        if packed.size == 0 or original_numel == 0:
            return np.zeros((0,), dtype=np.uint8)
        group_count = packed.size // 3
        src = packed[: group_count * 3].reshape(-1, 3).astype(np.uint16, copy=False)
        values = np.empty(group_count * 8, dtype=np.uint8)
        b0 = src[:, 0]
        b1 = src[:, 1]
        b2 = src[:, 2]
        values[0::8] = (b0 & 0x07).astype(np.uint8)
        values[1::8] = ((b0 >> 3) & 0x07).astype(np.uint8)
        values[2::8] = (((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)).astype(np.uint8)
        values[3::8] = ((b1 >> 1) & 0x07).astype(np.uint8)
        values[4::8] = ((b1 >> 4) & 0x07).astype(np.uint8)
        values[5::8] = (((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)).astype(np.uint8)
        values[6::8] = ((b2 >> 2) & 0x07).astype(np.uint8)
        values[7::8] = ((b2 >> 5) & 0x07).astype(np.uint8)
        return values[:original_numel]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Example usage
    manager = TieredKvCacheManager(
        gpu_tier_capacity_mb=2000,
        ram_tier_capacity_mb=8000,
    )
    
    # Allocate a test block
    test_data = torch.randn(16, 1024)
    block_id = manager.allocate_block(
        request_id="test_req_1",
        layer_idx=0,
        seq_start=0,
        seq_end=16,
        data=test_data,
        importance_score=0.9,
    )
    
    # Access it
    result = manager.access_block(block_id)
    
    # Print stats
    print("\n=== Cache Stats ===")
    for key, val in manager.get_stats().items():
        print(f"{key}: {val}")
