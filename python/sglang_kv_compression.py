"""
SGLang KV Compression Hook Integration

Provides encode/decode hooks for TurboQuant and other KV quantization backends.
Integrates with SGLang's attention forward path via the memory pool.

Architecture:
  prefill_forward() → ... → attention.forward() → save_kv() 
                                                    ↓
                                          _compress_kv_if_needed()
                                                    ↓
                                          store in KV pool (compressed)
                          
  decode_forward() → ... → attention.forward() → load_kv()
                                                    ↓
                                          _decompress_kv_if_needed()
                                                    ↓
                                          compute attention scores

This module provides:
  - TurboQuantizer encode/decode stubs (Phase 4.4.1-4.4.2)
  - KV pool allocator for compressed storage (Phase 4.4.3)
  - Telemetry hooks (Phase 4.4.4)
  - Feature gates and fallback chain (Phase 4.5)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

logger = logging.getLogger(__name__)


class CompressionMode(Enum):
    """Supported KV compression modes matching --kv-cache-dtype"""
    FP16 = "fp16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT4 = "int4"
    TQ1 = "tq1"
    TQ2 = "tq2"
    TQ3 = "tq3"
    TQ4 = "tq4"
    TQ8 = "tq8"
    RQ3_PLANAR = "rq3_planar"
    RQ4_PLANAR = "rq4_planar"
    RQ3_ISO = "rq3_iso"
    RQ4_ISO = "rq4_iso"


class CompressionStatus(Enum):
    """Status of a compression operation"""
    SUCCESS = "success"
    FALLBACK = "fallback"
    ERROR = "error"


@dataclass
class CompressedKV:
    """Container for compressed KV data"""
    data: torch.Tensor  # Compressed payload
    scale: Optional[torch.Tensor] = None  # Optional scale factors
    zero_point: Optional[torch.Tensor] = None  # Optional zero points
    mode: CompressionMode = CompressionMode.FP16
    compression_ratio: float = 1.0
    token_indices: Optional[torch.Tensor] = None  # Original token positions
    
    def size_bytes(self) -> int:
        """Return size in bytes including metadata"""
        total = self.data.numel() * self.data.element_size()
        if self.scale is not None:
            total += self.scale.numel() * self.scale.element_size()
        if self.zero_point is not None:
            total += self.zero_point.numel() * self.zero_point.element_size()
        return total


@dataclass
class CompressionTelemetry:
    """Telemetry for compression operations"""
    mode: CompressionMode
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    encode_us: int  # Encode time in microseconds
    decode_us: int  # Decode time in microseconds
    fallback_chain: list[CompressionMode]  # Attempted fallbacks
    status: CompressionStatus
    timestamp: float


class KVCompressionBackend(ABC):
    """Abstract base for KV compression backends"""

    @abstractmethod
    def encode(self, kv_data: torch.Tensor) -> CompressedKV:
        """Compress KV data into CompressedKV format"""
        pass

    @abstractmethod
    def decode(self, compressed: CompressedKV) -> torch.Tensor:
        """Decompress CompressedKV back to original tensor"""
        pass

    @abstractmethod
    def estimate_inner_product(
        self, q: torch.Tensor, compressed: CompressedKV
    ) -> torch.Tensor:
        """Estimate Q @ K^T using compressed K (for attention scores)"""
        pass

    @abstractmethod
    def get_compression_ratio(self) -> float:
        """Return expected compression ratio (bytes_compressed / bytes_original)"""
        pass

    @abstractmethod
    def supports_hardware(self) -> bool:
        """Check if this backend is supported on current hardware (gfx1030)"""
        pass

    @abstractmethod
    def get_mode(self) -> CompressionMode:
        """Return the compression mode this backend implements"""
        pass


class TurboQuantBackend(KVCompressionBackend):
    """
    TurboQuant KV compression backend (Phase 4.4.1-4.4.2 STUB)
    
    Compress K and V tensors using TurboQuant algorithm.
    This is a placeholder; real implementation in Phase 5.
    """

    def __init__(self, bit_width: int = 2, seed: int = 42, use_rope: bool = False):
        self.bit_width = bit_width  # 1-4 or 8
        self.seed = seed
        self.use_rope = use_rope
        self.compression_ratio = {1: 0.125, 2: 0.25, 3: 0.375, 4: 0.5, 8: 1.0}.get(
            bit_width, 0.25
        )

    def encode(self, kv_data: torch.Tensor) -> CompressedKV:
        """Encode KV data to TurboQuant format (STUB)"""
        start_us = time.perf_counter_ns() // 1000
        
        # STUB: Return identity encoding (no compression yet)
        # Phase 5 replaces this with real polar + QJL encoding
        compressed_data = kv_data.to(torch.float16)
        ratio = self.compression_ratio
        
        encode_us = (time.perf_counter_ns() // 1000) - start_us
        
        return CompressedKV(
            data=compressed_data,
            scale=None,
            zero_point=None,
            mode=CompressionMode(f"tq{self.bit_width}"),
            compression_ratio=ratio,
        )

    def decode(self, compressed: CompressedKV) -> torch.Tensor:
        """Decode TurboQuant data back to original format (STUB)"""
        start_us = time.perf_counter_ns() // 1000
        
        # STUB: Return identity decoding (no decompression yet)
        # Phase 5 replaces with real decompression
        original = compressed.data.to(torch.float32)
        
        decode_us = (time.perf_counter_ns() // 1000) - start_us
        
        return original

    def estimate_inner_product(
        self, q: torch.Tensor, compressed: CompressedKV
    ) -> torch.Tensor:
        """Estimate Q @ K^T using compressed K (STUB)"""
        # STUB: Use uncompressed K for now
        # Phase 5 replaces with fast approximate IP estimation
        k = self.decode(compressed)
        # Both q and k are [seq_len, heads, dim]
        # Return [seq_len_q, seq_len_k]
        return torch.matmul(q.mean(dim=1), k.mean(dim=1).t())

    def get_compression_ratio(self) -> float:
        return self.compression_ratio

    def supports_hardware(self) -> bool:
        # TODO: Check for AMD gfx1030 ROCm support
        return True

    def get_mode(self) -> CompressionMode:
        return CompressionMode(f"tq{self.bit_width}")


class RotorQuantBackend(KVCompressionBackend):
    """RotorQuant KV compression backend (future use)"""

    def __init__(self, bit_width: int = 3, mode: str = "planar"):
        self.bit_width = bit_width  # 3-4
        self.mode = mode  # "planar" or "iso"
        self.compression_ratio = {3: 0.375, 4: 0.5}.get(bit_width, 0.375)

    def encode(self, kv_data: torch.Tensor) -> CompressedKV:
        # STUB
        return CompressedKV(
            data=kv_data.to(torch.float16),
            mode=CompressionMode(f"rq{self.bit_width}_{self.mode}"),
            compression_ratio=self.compression_ratio,
        )

    def decode(self, compressed: CompressedKV) -> torch.Tensor:
        return compressed.data.to(torch.float32)

    def estimate_inner_product(
        self, q: torch.Tensor, compressed: CompressedKV
    ) -> torch.Tensor:
        k = self.decode(compressed)
        # Both q and k are [seq_len, heads, dim]
        # Return [seq_len_q, seq_len_k]
        return torch.matmul(q.mean(dim=1), k.mean(dim=1).t())

    def get_compression_ratio(self) -> float:
        return self.compression_ratio

    def supports_hardware(self) -> bool:
        return True

    def get_mode(self) -> CompressionMode:
        return CompressionMode(f"rq{self.bit_width}_{self.mode}")


class KVCompressionManager:
    """
    Manages KV compression for SGLang attention layers.
    
    Coordinates:
      - Backend selection and fallback chain
      - Encode/decode hooks in prefill and decode paths
      - KV pool allocator coordination
      - Telemetry collection
    """

    def __init__(
        self,
        mode: CompressionMode = CompressionMode.FP16,
        fallback_chain: Optional[list[CompressionMode]] = None,
        enable_telemetry: bool = True,
    ):
        self.mode = mode
        self.enable_telemetry = enable_telemetry
        self.telemetry_log: list[CompressionTelemetry] = []
        
        # Initialize backend
        self.backend = self._create_backend(mode)
        
        # Set fallback chain
        if fallback_chain is None:
            fallback_chain = [
                CompressionMode.TQ2,
                CompressionMode.TQ3,
                CompressionMode.TQ4,
                CompressionMode.FP8_E4M3,
                CompressionMode.FP16,
            ]
        self.fallback_chain = fallback_chain

    def _create_backend(self, mode: CompressionMode) -> Optional[KVCompressionBackend]:
        """Factory method to create appropriate backend"""
        if mode == CompressionMode.FP16:
            return None  # No compression
        elif mode in [CompressionMode.TQ1, CompressionMode.TQ2, 
                      CompressionMode.TQ3, CompressionMode.TQ4, 
                      CompressionMode.TQ8]:
            bit_width = int(mode.value[2])
            return TurboQuantBackend(bit_width=bit_width)
        elif mode in [CompressionMode.RQ3_PLANAR, CompressionMode.RQ4_PLANAR,
                      CompressionMode.RQ3_ISO, CompressionMode.RQ4_ISO]:
            parts = mode.value.split("_")
            bit_width = int(parts[0][2])
            rq_mode = parts[1]
            return RotorQuantBackend(bit_width=bit_width, mode=rq_mode)
        else:
            logger.warning(f"Unknown compression mode: {mode}, falling back to FP16")
            return None

    def encode_kv(
        self, k: torch.Tensor, v: torch.Tensor, layer_id: int
    ) -> Tuple[CompressedKV, CompressedKV]:
        """
        Encode K and V tensors (Phase 4.4.1 hook).
        
        Called after prefill attention in forward pass.
        """
        if self.backend is None:
            return CompressedKV(data=k, mode=self.mode), CompressedKV(data=v, mode=self.mode)
        
        try:
            k_compressed = self.backend.encode(k)
            v_compressed = self.backend.encode(v)
            
            if self.enable_telemetry:
                self._log_encode_telemetry(k, k_compressed, layer_id)
            
            return k_compressed, v_compressed
        except Exception as e:
            logger.warning(f"Encode failed, using fallback: {e}")
            return self._encode_with_fallback(k, v, layer_id)

    def decode_kv(
        self, k_compressed: CompressedKV, v_compressed: CompressedKV, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode compressed K and V (Phase 4.4.2 hook).
        
        Called before decode attention to retrieve original KV for scoring.
        """
        if self.backend is None:
            return k_compressed.data, v_compressed.data
        
        try:
            k = self.backend.decode(k_compressed)
            v = self.backend.decode(v_compressed)
            
            if self.enable_telemetry:
                self._log_decode_telemetry(k_compressed, k, layer_id)
            
            return k, v
        except Exception as e:
            logger.warning(f"Decode failed, using fallback: {e}")
            return self._decode_with_fallback(k_compressed, v_compressed, layer_id)

    def estimate_attention_scores(
        self, q: torch.Tensor, k_compressed: CompressedKV
    ) -> torch.Tensor:
        """
        Estimate Q @ K^T using compressed K (optional optimization).
        
        If backend supports approximate IP, use it; otherwise fall back to decode + matmul.
        """
        if self.backend is None:
            return torch.matmul(q, k_compressed.data.transpose(-2, -1))
        
        try:
            return self.backend.estimate_inner_product(q, k_compressed)
        except Exception as e:
            logger.warning(f"Approximate IP failed, using decode + matmul: {e}")
            k = self.backend.decode(k_compressed)
            return torch.matmul(q, k.transpose(-2, -1))

    def _encode_with_fallback(
        self, k: torch.Tensor, v: torch.Tensor, layer_id: int
    ) -> Tuple[CompressedKV, CompressedKV]:
        """Fallback chain for encoding"""
        for fallback_mode in self.fallback_chain:
            try:
                backend = self._create_backend(fallback_mode)
                if backend is None or backend.supports_hardware():
                    k_compressed = backend.encode(k) if backend else CompressedKV(data=k, mode=fallback_mode)
                    v_compressed = backend.encode(v) if backend else CompressedKV(data=v, mode=fallback_mode)
                    logger.info(f"Fallback to {fallback_mode.value} for layer {layer_id}")
                    return k_compressed, v_compressed
            except Exception:
                continue
        
        logger.error("All fallback modes failed, using uncompressed FP32")
        return CompressedKV(data=k, mode=CompressionMode.FP16), CompressedKV(data=v, mode=CompressionMode.FP16)

    def _decode_with_fallback(
        self, k_compressed: CompressedKV, v_compressed: CompressedKV, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback for decoding (usually return as-is if not compressed)"""
        return k_compressed.data, v_compressed.data

    def _log_encode_telemetry(
        self, original: torch.Tensor, compressed: CompressedKV, layer_id: int
    ):
        """Log compression telemetry (Phase 4.4.4)"""
        orig_bytes = original.numel() * original.element_size()
        comp_bytes = compressed.size_bytes()
        ratio = comp_bytes / orig_bytes if orig_bytes > 0 else 1.0
        
        # Append to telemetry log
        telemetry = CompressionTelemetry(
            mode=compressed.mode,
            original_bytes=orig_bytes,
            compressed_bytes=comp_bytes,
            compression_ratio=ratio,
            encode_us=0,  # TODO: measure actual encode time
            decode_us=0,  # TODO: measure actual decode time
            fallback_chain=[],
            status=CompressionStatus.SUCCESS,
            timestamp=time.time(),
        )
        self.telemetry_log.append(telemetry)
        
        logger.debug(
            f"Layer {layer_id} KV encode: "
            f"{orig_bytes:,} → {comp_bytes:,} bytes ({ratio:.2%} ratio), "
            f"mode={compressed.mode.value}"
        )

    def _log_decode_telemetry(
        self, compressed: CompressedKV, original: torch.Tensor, layer_id: int
    ):
        """Log decompression telemetry (Phase 4.4.4)"""
        comp_bytes = compressed.size_bytes()
        orig_bytes = original.numel() * original.element_size()
        ratio = comp_bytes / orig_bytes if orig_bytes > 0 else 1.0
        
        logger.debug(
            f"Layer {layer_id} KV decode: "
            f"{comp_bytes:,} → {orig_bytes:,} bytes ({ratio:.2%} ratio), "
            f"mode={compressed.mode.value}"
        )

    def get_stats(self) -> dict:
        """Return aggregate compression statistics"""
        if not self.telemetry_log:
            return {
                "total_original_bytes": 0,
                "total_compressed_bytes": 0,
                "average_ratio": 1.0,
                "events": 0,
                "mode": self.mode.value,
            }
        
        total_orig = sum(t.original_bytes for t in self.telemetry_log)
        total_comp = sum(t.compressed_bytes for t in self.telemetry_log)
        avg_ratio = sum(t.compression_ratio for t in self.telemetry_log) / len(self.telemetry_log)
        
        return {
            "total_original_bytes": total_orig,
            "total_compressed_bytes": total_comp,
            "average_ratio": avg_ratio,
            "events": len(self.telemetry_log),
            "mode": self.mode.value,
        }


# Global compression manager (Phase 4.5 feature gate)
_compression_manager: Optional[KVCompressionManager] = None


def init_kv_compression(
    mode: str = "fp16",
    fallback_chain: Optional[list[str]] = None,
    enable_telemetry: bool = True,
) -> KVCompressionManager:
    """
    Initialize global KV compression manager.
    
    Called once at server startup to set up compression for all layers.
    """
    global _compression_manager
    
    try:
        compression_mode = CompressionMode(mode)
    except ValueError:
        logger.error(f"Invalid compression mode: {mode}, using fp16")
        compression_mode = CompressionMode.FP16
    
    fallback = None
    if fallback_chain:
        fallback = [CompressionMode(m) for m in fallback_chain]
    
    _compression_manager = KVCompressionManager(
        mode=compression_mode,
        fallback_chain=fallback,
        enable_telemetry=enable_telemetry,
    )
    
    logger.info(f"KV compression initialized: {compression_mode.value}")
    return _compression_manager


def get_kv_compression_manager() -> Optional[KVCompressionManager]:
    """Get the global compression manager (or None if not initialized)"""
    return _compression_manager


def compress_kv_if_enabled(
    k: torch.Tensor, v: torch.Tensor, layer_id: int
) -> Tuple[CompressedKV, CompressedKV]:
    """Wrapper for encode hook that respects feature gate"""
    if _compression_manager is None:
        return CompressedKV(data=k, mode=CompressionMode.FP16), CompressedKV(data=v, mode=CompressionMode.FP16)
    return _compression_manager.encode_kv(k, v, layer_id)


def decompress_kv_if_needed(
    k_compressed: CompressedKV, v_compressed: CompressedKV, layer_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wrapper for decode hook that respects feature gate"""
    if _compression_manager is None:
        return k_compressed.data, v_compressed.data
    return _compression_manager.decode_kv(k_compressed, v_compressed, layer_id)
