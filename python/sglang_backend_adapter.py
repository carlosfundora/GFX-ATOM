"""
SGLang TurboQuant Backend Adapter.

Bridges SGLang codec flags (--kv-cache-dtype tq2) to gfxATOM TurboQuantizer.

This adapter:
1. Receives SGLang ServerArgs with kv_cache_dtype set
2. Resolves to gfxATOM KvCodec enum
3. Instantiates TurboQuantizer with correct bit width
4. Provides encode/decode/estimate_inner_product interface
5. Handles fallback to Triton if needed
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import torch

from kv_codec_adapters import CodecAdapterRegistry, CodecBackendPlan
from sglang_autoquant_bridge import (
    AutoQuantBackendSummary,
    AutoQuantPolicySnapshot,
    build_autoquant_backend_summary,
)
from kv_quant_contracts import KvCodec, normalize_codec_alias
from tiered_kv_cache_manager import TieredKvCacheManager, CacheTier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SGLangCodecConfig:
    """Configuration resolved from SGLang --kv-cache-dtype flag."""
    flag_value: str  # raw string like "tq2"
    codec: KvCodec  # resolved enum
    family: str
    bit_width: int
    compression_ratio: float
    is_experimental: bool
    fallback_enabled: bool


# Codec → configuration mapping
SGLANG_CODEC_CONFIGS = {
    KvCodec.tq1: SGLangCodecConfig(
        flag_value="tq1",
        codec=KvCodec.tq1,
        family="turbo",
        bit_width=1,
        compression_ratio=16.0,
        is_experimental=True,
        fallback_enabled=True,
    ),
    KvCodec.tq2: SGLangCodecConfig(
        flag_value="tq2",
        codec=KvCodec.tq2,
        family="turbo",
        bit_width=2,
        compression_ratio=8.0,
        is_experimental=False,
        fallback_enabled=True,
    ),
    KvCodec.tq3: SGLangCodecConfig(
        flag_value="tq3",
        codec=KvCodec.tq3,
        family="turbo",
        bit_width=3,
        compression_ratio=5.33,
        is_experimental=False,
        fallback_enabled=True,
    ),
    KvCodec.tq4: SGLangCodecConfig(
        flag_value="tq4",
        codec=KvCodec.tq4,
        family="turbo",
        bit_width=4,
        compression_ratio=4.0,
        is_experimental=False,
        fallback_enabled=True,
    ),
    KvCodec.tq8: SGLangCodecConfig(
        flag_value="tq8",
        codec=KvCodec.tq8,
        family="turbo",
        bit_width=8,
        compression_ratio=2.0,
        is_experimental=False,
        fallback_enabled=False,
    ),
    KvCodec.rq3_planar: SGLangCodecConfig(
        flag_value="rq3_planar",
        codec=KvCodec.rq3_planar,
        family="rotor_planar",
        bit_width=3,
        compression_ratio=5.33,
        is_experimental=False,
        fallback_enabled=True,
    ),
    KvCodec.rq4_planar: SGLangCodecConfig(
        flag_value="rq4_planar",
        codec=KvCodec.rq4_planar,
        family="rotor_planar",
        bit_width=4,
        compression_ratio=4.0,
        is_experimental=False,
        fallback_enabled=True,
    ),
    KvCodec.rq3_iso: SGLangCodecConfig(
        flag_value="rq3_iso",
        codec=KvCodec.rq3_iso,
        family="rotor_iso",
        bit_width=3,
        compression_ratio=5.33,
        is_experimental=False,
        fallback_enabled=True,
    ),
    KvCodec.rq4_iso: SGLangCodecConfig(
        flag_value="rq4_iso",
        codec=KvCodec.rq4_iso,
        family="rotor_iso",
        bit_width=4,
        compression_ratio=4.0,
        is_experimental=False,
        fallback_enabled=True,
    ),
}


def resolve_sglang_codec(kv_cache_dtype_flag: str) -> Optional[SGLangCodecConfig]:
    """
    Resolve SGLang --kv-cache-dtype flag to gfxATOM codec configuration.
    
    Args:
        kv_cache_dtype_flag: String like "tq2", "auto", "fp8_e4m3", etc.
    
    Returns:
        SGLangCodecConfig if it's a supported quantized KV mode, None for native SGLang modes
    
    Raises:
        ValueError: If flag is invalid
    """
    try:
        codec = normalize_codec_alias(kv_cache_dtype_flag)
    except ValueError:
        logger.error(f"Invalid kv_cache_dtype flag: {kv_cache_dtype_flag}")
        raise
    
    # Return config if it's a supported quantized KV mode
    return SGLANG_CODEC_CONFIGS.get(codec)


class SGLangTurboQuantAdapter:
    """
    Adapter bridging SGLang TurboQuant codec selection to gfxATOM TurboQuantizer.
    
    Usage:
        # In SGLang model executor init:
        adapter = SGLangTurboQuantAdapter(
            kv_cache_dtype_flag="tq2",
            dimension=256,
            num_heads=32,
        )
        
        # In prefill forward:
        turbo_code = adapter.encode_kv(k_cache)  # [batch, n_tokens, d] → TurboCode
        
        # In decode forward:
        scores = adapter.estimate_inner_product(turbo_code, query)  # TurboCode → scores
    """
    
    def __init__(
        self,
        kv_cache_dtype_flag: str,
        dimension: int,
        num_heads: int,
        num_projections: int = 128,  # default for QJL sketch
        enable_rope_quant: bool = True,  # respect SGLANG_KV_CACHE_TURBOQUANT_ROPE
        enable_qjl: bool = True,  # respect SGLANG_KV_CACHE_TURBOQUANT_QJL
        autoquant_policy: Optional[AutoQuantPolicySnapshot] = None,
    ):
        """
        Initialize adapter with SGLang codec flag.
        
        Args:
            kv_cache_dtype_flag: From --kv-cache-dtype (e.g., "tq2")
            dimension: Hidden dimension
            num_heads: Number of attention heads
            num_projections: Size of QJL sketch (default 128)
            enable_rope_quant: Use RoPE quantization (disable for MLA)
            enable_qjl: Enable QJL unbiased inner product
        """
        self.kv_cache_dtype_flag = kv_cache_dtype_flag
        self.dimension = dimension
        self.num_heads = num_heads
        self.num_projections = num_projections
        self.enable_rope_quant = enable_rope_quant
        self.enable_qjl = enable_qjl
        self.autoquant_policy = autoquant_policy
        self.autoquant_summary: AutoQuantBackendSummary | None = None
        self.registry = CodecAdapterRegistry()
        self.backend_plan: CodecBackendPlan | None = None
        
        # Resolve codec
        self.codec_config = resolve_sglang_codec(kv_cache_dtype_flag)
        if self.codec_config is None:
            # Native SGLang mode (auto, fp8_e4m3, etc.)
            logger.info(
                f"Native SGLang KV codec: {kv_cache_dtype_flag}. "
                f"Not using gfxATOM quant backend."
            )
            self.is_turbo_quant = False
            self.quant_family = "native"
            self.turboquant = None
            self.backend_chain = ("native",)
            return
        
        self.is_turbo_quant = True
        self.quant_family = self.codec_config.family
        
        # Warn if experimental
        if self.codec_config.is_experimental:
            logger.warning(
                f"⚠️  Quant mode '{kv_cache_dtype_flag}' is EXPERIMENTAL. "
                f"Use SGLANG_KV_CACHE_TURBOQUANT_QJL=1 for unbiased inner products."
            )
        
        logger.info(
            f"✓ Initialized SGLangTurboQuantAdapter: "
            f"codec={self.codec_config.codec.value}, "
            f"family={self.codec_config.family}, "
            f"bit_width={self.codec_config.bit_width}, "
            f"compression={self.codec_config.compression_ratio:.1f}x, "
            f"dimension={dimension}"
        )
        
        self._create_turboquant()
        self.backend_chain = self.resolve_backend_chain()
    
    def _create_turboquant(self):
        """
        Resolve the TurboQuant backend plan and preserve fallback metadata.
        
        In Phase 5, this will create a real TurboQuantizer instance
        with the resolved bit_width and configuration.
        """
        self.backend_plan = self.registry.backend_plan_for(self.codec_config.codec)
        if self.autoquant_policy is not None:
            self.autoquant_summary = build_autoquant_backend_summary(
                self.autoquant_policy,
                registry=self.registry,
            )
        self.turboquant = None
        if self.backend_plan is None:
            logger.info(
                "Quant codec %s has no backend plan; using native SGLang path.",
                self.codec_config.codec.value,
            )
            return

        logger.info(
            "Quant backend plan: preferred=%s fallback=%s ultimate=%s",
            self.backend_plan.preferred_backend,
            self.backend_plan.fallback_backend,
            self.backend_plan.ultimate_fallback,
        )

    def resolve_backend_chain(
        self,
        turboquant_available: bool = True,
        triton_available: bool = True,
    ) -> tuple[str, ...]:
        """Return the backend chain used for dispatch decisions."""
        if not self.is_turbo_quant:
            return ("native",)

        if self.backend_plan is None:
            return ("native",)

        chain = []
        if turboquant_available and self.backend_plan.supported:
            chain.append(self.backend_plan.preferred_backend)
        if not turboquant_available or not self.backend_plan.supported:
            if triton_available:
                chain.append(self.backend_plan.fallback_backend)
            chain.append(self.backend_plan.ultimate_fallback)
        elif triton_available:
            chain.extend([self.backend_plan.fallback_backend, self.backend_plan.ultimate_fallback])
        return tuple(dict.fromkeys(chain))

    def resolve_backend(
        self,
        turboquant_available: bool = True,
        triton_available: bool = True,
    ) -> str:
        """Select the first available backend from the dispatch chain."""
        chain = self.resolve_backend_chain(
            turboquant_available=turboquant_available,
            triton_available=triton_available,
        )
        return chain[0] if chain else "native"
    
    def is_enabled(self) -> bool:
        """Check if TurboQuant is enabled (vs. native SGLang mode)."""
        return self.is_turbo_quant
    
    def encode_kv(self, k_cache: torch.Tensor) -> "TurboCode":
        """
        Compress K cache using TurboQuantizer.
        
        Args:
            k_cache: K tensor, shape [batch*n_tokens, num_heads, head_dim]
        
        Returns:
            TurboCode with compressed polar_code and residual_sketch
        
        Note:
            In Phase 5, this will use real TurboQuantizer.encode().
            For Phase 4, returns a placeholder that preserves tensor shape.
        """
        if not self.is_turbo_quant:
            raise RuntimeError("TurboQuant not enabled; use native SGLang for encoding")
        
        # TODO: Call self.turboquant.encode(k_cache)
        # For now, placeholder:
        logger.debug(f"[placeholder] encode_kv: k_cache shape {k_cache.shape}")
        
        # Return mock TurboCode (Phase 5 replaces with real implementation)
        from dataclasses import dataclass
        @dataclass
        class MockTurboCode:
            polar_code: bytes
            residual_sketch: bytes
        
        return MockTurboCode(
            polar_code=b"mock_polar",
            residual_sketch=b"mock_qjl",
        )
    
    def estimate_inner_product(
        self,
        turbo_code: "TurboCode",
        query: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate attention scores from compressed TurboCode.
        
        Args:
            turbo_code: Compressed KV from encode_kv()
            query: Query tensor, shape [batch, num_heads, head_dim]
        
        Returns:
            Estimated inner product scores, shape [batch, n_compressed_tokens]
        
        Note:
            Uses unbiased inner product estimation if enable_qjl=True.
        """
        if not self.is_turbo_quant:
            raise RuntimeError("TurboQuant not enabled; cannot estimate inner products")
        
        # TODO: Call self.turboquant.estimate_inner_product(turbo_code, query)
        # For now, placeholder:
        logger.debug(f"[placeholder] estimate_inner_product: query shape {query.shape}")
        
        # Return mock scores (Phase 5 replaces with real implementation)
        batch_size = query.shape[0]
        n_tokens = 128  # mock value
        return torch.zeros(batch_size, n_tokens, device=query.device, dtype=query.dtype)
    
    def get_config_dict(self) -> dict:
        """Export configuration for logging/telemetry."""
        return {
            "flag": self.kv_cache_dtype_flag,
            "is_turbo_quant": self.is_turbo_quant,
            "quant_family": getattr(self, "quant_family", "native"),
            "codec": self.codec_config.codec.value if self.codec_config else None,
            "bit_width": self.codec_config.bit_width if self.codec_config else None,
            "compression_ratio": self.codec_config.compression_ratio if self.codec_config else None,
            "is_experimental": self.codec_config.is_experimental if self.codec_config else None,
            "enable_rope_quant": self.enable_rope_quant,
            "enable_qjl": self.enable_qjl,
            "dimension": self.dimension,
            "num_heads": self.num_heads,
            "num_projections": self.num_projections,
            "backend_chain": list(self.backend_chain),
            "backend_plan": (
                {
                    "codec": self.backend_plan.codec.value,
                    "family": self.backend_plan.family,
                    "preferred_backend": self.backend_plan.preferred_backend,
                    "fallback_backend": self.backend_plan.fallback_backend,
                    "ultimate_fallback": self.backend_plan.ultimate_fallback,
                    "supported": self.backend_plan.supported,
                    "bit_width": self.backend_plan.bit_width,
                    "is_experimental": self.backend_plan.is_experimental,
                }
                if self.backend_plan is not None
                else None
            ),
            "autoquant_summary": (
                self.autoquant_summary.to_dict()
                if self.autoquant_summary is not None
                else None
            ),
        }


def create_sglang_adapter_from_args(sglang_args, dimension: int, num_heads: int) -> SGLangTurboQuantAdapter:
    """
    Factory function to create adapter from SGLang ServerArgs.
    
    Args:
        sglang_args: ServerArgs dataclass from sglang.srt.server_args
        dimension: Hidden dimension
        num_heads: Number of attention heads
    
    Returns:
        Configured SGLangTurboQuantAdapter instance
    
    Example:
        from sglang.srt.server_args import ServerArgs
        
        args = ServerArgs(...)
        adapter = create_sglang_adapter_from_args(
            sglang_args=args,
            dimension=256,
            num_heads=32,
        )
        
        if adapter.is_enabled():
            turbo_code = adapter.encode_kv(k_cache)
    """
    return SGLangTurboQuantAdapter(
        kv_cache_dtype_flag=sglang_args.kv_cache_dtype,
        dimension=dimension,
        num_heads=num_heads,
    )


class SGLangRotorQuantAdapter:
    """
    Adapter bridging SGLang RotorQuant codec selection to gfxATOM RotorQuantizer.
    
    RotorQuant offers 28-35% decode speedup vs TurboQuant through reduced FMA count:
    - PlanarQuant: 64x fewer FMAs (2D Givens rotations)
    - IsoQuant: 32x fewer FMAs (4D quaternion rotations)
    
    Usage:
        adapter = SGLangRotorQuantAdapter(
            kv_cache_dtype_flag="rq3_planar",
            dimension=256,
            num_heads=32,
        )
        
        # Prefill: compress KV
        rotor_code = adapter.encode_kv(k_cache)
        
        # Decode: estimate attention scores
        scores = adapter.estimate_inner_product(rotor_code, query)
    """
    
    def __init__(
        self,
        kv_cache_dtype_flag: str,
        dimension: int,
        num_heads: int,
        num_layers: int = 32,
        seed: int = 42,
        autoquant_policy: Optional[AutoQuantPolicySnapshot] = None,
    ):
        """
        Initialize RotorQuant adapter.
        
        Args:
            kv_cache_dtype_flag: From --kv-cache-dtype (e.g., "rq3_planar", "rq3_iso")
            dimension: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            seed: Random seed for reproducible rotation generation
            autoquant_policy: Optional AutoQuant policy for backend selection
        """
        self.kv_cache_dtype_flag = kv_cache_dtype_flag
        self.dimension = dimension
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seed = seed
        self.autoquant_policy = autoquant_policy
        self._rust_failed = False
        
        # Resolve codec config
        self.codec_config = resolve_sglang_codec(kv_cache_dtype_flag)
        if self.codec_config is None:
            raise ValueError(f"RotorQuant not supported for: {kv_cache_dtype_flag}")
        
        # Determine rotation mode
        self.is_planar = "planar" in self.codec_config.family
        self.is_iso = "iso" in self.codec_config.family
        self._rust_mode = self._resolve_rust_mode()
        self._rust_codec = self._init_rust_codec()
        
        logger.info(
            f"RotorQuantAdapter: mode={kv_cache_dtype_flag}, "
            f"dim={dimension}, heads={num_heads}, "
            f"planar={self.is_planar}, iso={self.is_iso}"
        )
    
    def encode_kv(
        self,
        k_cache: torch.Tensor,  # [batch, n_tokens, dim]
        v_cache: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Encode KV cache using RotorQuant compression.
        
        Applies rotation-based quantization for 28-35% decode speedup.
        
        Args:
            k_cache: Key cache tensor [batch, n_tokens, dim]
            v_cache: Value cache tensor (optional, same shape as k_cache)
        
        Returns:
            Dict with compressed codes and metadata
        """
        batch_size, n_tokens, _ = k_cache.shape
        bit_width = self.codec_config.bit_width

        # Flatten to [batch*n_tokens, dim] and normalize row-wise for stable codec behavior.
        k_flat = k_cache.reshape(-1, self.dimension).float().cpu()
        scales = k_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = (k_flat / scales).clamp(-1.0, 1.0)

        backend = "fallback_quantized"
        compressed_k: torch.Tensor
        row_lengths: torch.Tensor | None = None
        quant_levels = (1 << bit_width) - 1

        # Rust-first hot path: only fallback after an actual rust encode failure.
        if self._rust_codec is not None and not self._rust_failed:
            try:
                compressed_k, row_lengths = self._encode_rust_rows(normalized)
                backend = "rust"
            except Exception as exc:
                self._rust_failed = True
                logger.warning(
                    "RotorQuant rust encode failed; falling back for %s: %s",
                    self.kv_cache_dtype_flag,
                    exc,
                )
                compressed_k = self._encode_fallback(normalized, quant_levels)
        else:
            compressed_k = self._encode_fallback(normalized, quant_levels)

        return {
            "compressed_k": compressed_k,
            "bit_width": bit_width,
            "n_tokens": n_tokens,
            "batch_size": batch_size,
            "dimension": self.dimension,
            "dtype_flag": self.kv_cache_dtype_flag,
            "is_planar": self.is_planar,
            "is_iso": self.is_iso,
            "seed": self.seed,
            "backend": backend,
            "rust_path_preferred": True,
            "rust_path_failed": self._rust_failed,
            "row_lengths": row_lengths,
            "quant_levels": quant_levels,
            "scales": scales.squeeze(1),
        }
    
    def estimate_inner_product(
        self,
        rotor_code: dict,
        query: torch.Tensor,  # [batch, dim]
    ) -> torch.Tensor:
        """
        Estimate inner product scores from RotorQuant-compressed KV.
        
        Provides unbiased inner product estimation like TurboQuant, but with
        28-35% lower computational cost due to rotation-based design.
        
        Args:
            rotor_code: Compressed KV from encode_kv()
            query: Query tensor [batch, dim]
        
        Returns:
            Attention scores [batch, n_tokens]
        """
        n_tokens = rotor_code["n_tokens"]
        batch_size = query.shape[0]
        decompressed = self._decode_k(rotor_code).to(device=query.device, dtype=query.dtype)
        decompressed = decompressed.reshape(batch_size, n_tokens, self.dimension)
        return torch.einsum("bd,btd->bt", query, decompressed)

    def _resolve_rust_mode(self) -> str:
        if self.is_planar and self.codec_config.bit_width == 3:
            return "planar3"
        if self.is_planar and self.codec_config.bit_width == 4:
            return "planar4"
        if self.is_iso and self.codec_config.bit_width == 3:
            return "iso3"
        if self.is_iso and self.codec_config.bit_width == 4:
            return "iso4"
        raise ValueError(f"Unsupported RotorQuant mode: {self.kv_cache_dtype_flag}")

    def _init_rust_codec(self):
        try:
            from rs_rotorquant_codec import PyRotorQuantCodec  # type: ignore

            return PyRotorQuantCodec(self._rust_mode, self.seed, self.is_planar)
        except Exception as exc:
            self._rust_failed = True
            logger.warning(
                "Rust RotorQuant module unavailable for %s, fallback path active: %s",
                self.kv_cache_dtype_flag,
                exc,
            )
            return None

    def _encode_rust_rows(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._rust_codec is not None
        encoded_rows: list[torch.Tensor] = []
        row_lengths: list[int] = []
        for row in normalized:
            row_data = row.tolist()
            if self.is_planar:
                compressed = self._rust_codec.compress_planar(row_data, self.dimension)
            else:
                compressed = self._rust_codec.compress_iso(row_data, self.dimension)
            encoded = torch.tensor(compressed, dtype=torch.uint8)
            encoded_rows.append(encoded)
            row_lengths.append(int(encoded.numel()))
        max_len = max(row_lengths) if row_lengths else 0
        padded = torch.zeros((len(encoded_rows), max_len), dtype=torch.uint8)
        for i, row in enumerate(encoded_rows):
            padded[i, : row.numel()] = row
        return padded, torch.tensor(row_lengths, dtype=torch.int32)

    def _encode_fallback(self, normalized: torch.Tensor, quant_levels: int) -> torch.Tensor:
        return torch.round((normalized + 1.0) * 0.5 * quant_levels).to(torch.uint8)

    def _decode_k(self, rotor_code: dict) -> torch.Tensor:
        backend = rotor_code.get("backend", "fallback_quantized")
        compressed_k = rotor_code["compressed_k"]
        scales = rotor_code["scales"].float().cpu()
        if backend == "rust" and self._rust_codec is not None and not self._rust_failed:
            try:
                row_lengths = rotor_code.get("row_lengths")
                decoded_rows: list[torch.Tensor] = []
                for idx in range(compressed_k.shape[0]):
                    row = compressed_k[idx]
                    length = int(row_lengths[idx]) if row_lengths is not None else int(row.numel())
                    row_bytes = row[:length].tolist()
                    if self.is_planar:
                        decoded = self._rust_codec.decompress_planar(row_bytes, self.dimension)
                    else:
                        decoded = self._rust_codec.decompress_iso(row_bytes, self.dimension)
                    decoded_rows.append(torch.tensor(decoded, dtype=torch.float32))
                normalized = torch.stack(decoded_rows, dim=0)
                return normalized * scales.unsqueeze(1)
            except Exception as exc:
                self._rust_failed = True
                logger.warning(
                    "RotorQuant rust decode failed; falling back for %s: %s",
                    self.kv_cache_dtype_flag,
                    exc,
                )

        quant_levels = float(rotor_code.get("quant_levels", (1 << rotor_code["bit_width"]) - 1))
        normalized = (compressed_k.float().cpu() / quant_levels) * 2.0 - 1.0
        return normalized * scales.unsqueeze(1)
    
    def fallback_to_turbo(self) -> Optional['SGLangTurboQuantAdapter']:
        """
        Fallback to TurboQuant if RotorQuant unavailable.
        
        Selection logic:
        - rq3_planar → tq2 (similar 3-bit granularity)
        - rq4_planar → tq4 (similar 4-bit granularity)
        - rq3_iso → tq2
        - rq4_iso → tq4
        
        Returns:
            SGLangTurboQuantAdapter instance, or None if fallback disabled
        """
        if not self.codec_config.fallback_enabled:
            return None
        
        fallback_map = {
            "rq3_planar": "tq2",
            "rq4_planar": "tq4",
            "rq3_iso": "tq2",
            "rq4_iso": "tq4",
        }
        
        fallback_flag = fallback_map.get(self.kv_cache_dtype_flag)
        if fallback_flag is None:
            return None
        
        logger.warning(f"Falling back from {self.kv_cache_dtype_flag} to {fallback_flag}")
        
        return SGLangTurboQuantAdapter(
            kv_cache_dtype_flag=fallback_flag,
            dimension=self.dimension,
            num_heads=self.num_heads,
            autoquant_policy=self.autoquant_policy,
        )


class CompressionDispatcher:
    """
    Intelligent KV cache compression dispatcher.
    
    Selects between TurboQuant and RotorQuant based on:
    - Model characteristics (long-context → prefer RQ)
    - Hardware capabilities (CUDA → RQ with Triton; CPU/ROCm → fallback to TQ)
    - User preference (explicit flag overrides heuristics)
    - Available VRAM (low VRAM → prefer higher compression ratio)
    """
    
    def __init__(
        self,
        user_preference: Optional[str] = None,  # explicit --kv-cache-dtype flag
        model_type: Optional[str] = None,  # "long-context", "short-context", etc.
        available_vram_mb: float = 12000.0,  # typical gfx1030: 12GB
    ):
        """
        Initialize compression dispatcher.
        
        Args:
            user_preference: User's explicit codec choice (overrides heuristics)
            model_type: Model characteristics for heuristic selection
            available_vram_mb: Available GPU VRAM in MB
        """
        self.user_preference = user_preference
        self.model_type = model_type
        self.available_vram_mb = available_vram_mb
    
    def select_codec(
        self,
        dimension: int,
        num_heads: int,
        max_seq_len: int,
    ) -> str:
        """
        Select best compression codec for given model dimensions.
        
        Args:
            dimension: Hidden dimension
            num_heads: Attention heads
            max_seq_len: Maximum sequence length
        
        Returns:
            Codec flag string (e.g., "rq3_planar", "tq2")
        """
        if self.user_preference:
            return self.user_preference
        
        # Heuristic: prefer RotorQuant for long-context (28-35% faster decode)
        is_long_context = max_seq_len >= 4096
        
        if is_long_context:
            # Prefer RQ for long-context; fallback to TQ if RQ unavailable
            return "rq3_planar"
        else:
            # TQ2 for short-context (simpler, wider compatibility)
            return "tq2"
    
    def create_adapter(
        self,
        codec_flag: str,
        dimension: int,
        num_heads: int,
    ):
        """
        Create appropriate adapter (RotorQuant or TurboQuant).
        
        Args:
            codec_flag: Selected codec flag
            dimension: Hidden dimension
            num_heads: Attention heads
        
        Returns:
            SGLangRotorQuantAdapter or SGLangTurboQuantAdapter instance
        """
        if "rq" in codec_flag:
            return SGLangRotorQuantAdapter(
                kv_cache_dtype_flag=codec_flag,
                dimension=dimension,
                num_heads=num_heads,
            )
        else:
            return SGLangTurboQuantAdapter(
                kv_cache_dtype_flag=codec_flag,
                dimension=dimension,
                num_heads=num_heads,
            )


class TieredKvCacheAdapter:
    """
    Two-tier KV cache adapter: RotorQuant (GPU) + TurboQuant (RAM spill).
    
    Strategy:
    - **Tier 1 (GPU)**: RotorQuant with 3-bit quantization
      - 16-value blocks, 2 bytes metadata (rotation index + scale)
      - Per 64 bytes (16×float32): ~6 bytes + 2 bytes metadata = 8 bytes total (8x compression)
      - Fast GPU access, primary tier for hot/recent sequences
    
    - **Tier 2 (RAM)**: TurboQuant KV cache (spill for older/cold sequences)
      - Moved to system RAM when GPU tier full
      - Called back on-demand with minimal latency penalty
      - Useful for extremely long-context scenarios (e.g., 100K+ tokens)
    
    Features:
    - Adaptive eviction: LRU + importance weighting
    - Block-level granularity for fine-grained swapping
    - Metrics: hit rate, swap overhead, space efficiency
    - Automatic promotion of hot blocks from RAM to GPU
    
    Usage:
        adapter = TieredKvCacheAdapter(
            gpu_capacity_mb=8000,
            ram_capacity_mb=32000,
        )
        
        # Allocate blocks (automatic tier assignment)
        block_id = adapter.allocate_kv_block(
            request_id="req_1",
            layer_idx=0,
            k_cache=k_tensor,
            v_cache=v_tensor,
            importance_score=0.9,  # importance-weighted attention
        )
        
        # Access blocks (automatic promotion if hot)
        k_decompressed = adapter.get_kv_block(block_id)
        
        # Monitor cache behavior
        stats = adapter.get_cache_stats()
    """
    
    def __init__(
        self,
        gpu_capacity_mb: int = 8000,
        ram_capacity_mb: int = 32000,
        primary_codec: str = "rq3_planar",
        secondary_codec: str = "tq2",
        dimension: int = 4096,
        num_heads: int = 32,
    ):
        """
        Initialize tiered KV cache adapter.
        
        Args:
            gpu_capacity_mb: GPU tier capacity in MB (Tier 1, RotorQuant)
            ram_capacity_mb: RAM tier capacity in MB (Tier 2, TurboQuant spill)
            primary_codec: Primary codec (e.g., "rq3_planar" for RotorQuant)
            secondary_codec: Secondary codec (e.g., "tq2" for TurboQuant)
            dimension: Hidden dimension for codecs
            num_heads: Attention heads for codecs
        """
        self.gpu_capacity_mb = gpu_capacity_mb
        self.ram_capacity_mb = ram_capacity_mb
        self.primary_codec = primary_codec
        self.secondary_codec = secondary_codec
        self.dimension = dimension
        self.num_heads = num_heads
        
        # Underlying tiered cache manager
        self.cache_mgr = TieredKvCacheManager(
            gpu_tier_capacity_mb=gpu_capacity_mb,
            ram_tier_capacity_mb=ram_capacity_mb,
        )
        
        # Codec adapters
        self.primary_adapter = self._create_adapter(primary_codec)
        self.secondary_adapter = self._create_adapter(secondary_codec)
        
        logger.info(
            f"TieredKvCacheAdapter initialized: "
            f"GPU={gpu_capacity_mb}MB ({primary_codec}), "
            f"RAM={ram_capacity_mb}MB ({secondary_codec}), "
            f"dim={dimension}, heads={num_heads}"
        )
    
    def allocate_kv_block(
        self,
        request_id: str,
        layer_idx: int,
        k_cache: torch.Tensor,
        v_cache: Optional[torch.Tensor] = None,
        importance_score: float = 1.0,
    ) -> int:
        """
        Allocate a new KV block to the appropriate tier.
        
        Args:
            request_id: Request ID for tracking
            layer_idx: Transformer layer index
            k_cache: Key cache tensor [seq_len, dim]
            v_cache: Value cache tensor (optional, same shape as k_cache)
            importance_score: Importance weight for eviction decisions (0-1)
        
        Returns:
            Block ID for future access
        """
        # Combine K and V for storage (if V provided)
        combined = k_cache if v_cache is None else torch.cat([k_cache, v_cache], dim=-1)
        
        # Get sequence boundaries
        seq_len = combined.shape[0]
        seq_start = 0
        seq_end = seq_len
        
        # Allocate in tiered manager (automatically selects tier)
        block_id = self.cache_mgr.allocate_block(
            request_id=request_id,
            layer_idx=layer_idx,
            seq_start=seq_start,
            seq_end=seq_end,
            data=combined,
            importance_score=importance_score,
        )
        
        logger.debug(
            f"Allocated KV block {block_id}: "
            f"req={request_id}, layer={layer_idx}, "
            f"importance={importance_score:.2f}"
        )
        
        return block_id
    
    def get_kv_block(self, block_id: int) -> torch.Tensor:
        """
        Retrieve a KV block, with automatic promotion if hot.
        
        Args:
            block_id: Block ID from allocate_kv_block()
        
        Returns:
            Decompressed KV tensor
        """
        return self.cache_mgr.access_block(block_id)
    
    def evict_block(self, block_id: int) -> None:
        """Explicitly evict a block from cache."""
        self.cache_mgr.evict_block(block_id)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get tiered cache statistics.
        
        Returns:
            Dict with GPU tier, RAM tier, and aggregate stats
        """
        return self.cache_mgr.get_stats()
    
    def print_cache_summary(self) -> None:
        """Print human-readable cache summary."""
        stats = self.get_cache_stats()
        
        gpu_stats = stats["gpu_tier"]
        ram_stats = stats["ram_tier"]
        
        logger.info(
            f"\n=== Tiered KV Cache Summary ===\n"
            f"GPU Tier (RotorQuant):\n"
            f"  Blocks: {gpu_stats['blocks']}\n"
            f"  Used: {gpu_stats['bytes'] / 1024 / 1024:.1f}MB / {gpu_stats['capacity_bytes'] / 1024 / 1024:.1f}MB "
            f"({gpu_stats['utilization_pct']:.1f}%)\n"
            f"  Hit Rate: {gpu_stats['hit_rate']:.1%}\n"
            f"\nRAM Tier (TurboQuant spill):\n"
            f"  Blocks: {ram_stats['blocks']}\n"
            f"  Used: {ram_stats['bytes'] / 1024 / 1024:.1f}MB / {ram_stats['capacity_bytes'] / 1024 / 1024:.1f}MB "
            f"({ram_stats['utilization_pct']:.1f}%)\n"
            f"  Misses (RAM accesses): {ram_stats['misses']}\n"
            f"  Promotions (RAM→GPU): {ram_stats['swaps_in']}\n"
            f"  Demotions (GPU→RAM): {ram_stats['swaps_out']}\n"
            f"\nCompression: {stats['compression_ratio']:.1f}x ({stats['block_size']}-value blocks)"
        )
    
    # ============ Private Helpers ============
    
    def _create_adapter(self, codec_flag: str):
        """Create a codec adapter (RotorQuant or TurboQuant)."""
        if "rq" in codec_flag:
            return SGLangRotorQuantAdapter(
                kv_cache_dtype_flag=codec_flag,
                dimension=self.dimension,
                num_heads=self.num_heads,
            )
        else:
            return SGLangTurboQuantAdapter(
                kv_cache_dtype_flag=codec_flag,
                dimension=self.dimension,
                num_heads=self.num_heads,
            )
