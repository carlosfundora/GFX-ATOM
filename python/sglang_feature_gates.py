"""
Phase 4.5: Feature Gates and Production Safety

Provides production-safe configuration and fallback mechanisms for KV compression:
  - Feature gating (experimental modes off by default)
  - Runtime safety checks (hardware detection, config validation)
  - Graceful fallback chains
  - Comprehensive error logging and user guidance

This module integrates sglang_kv_compression and sglang_kv_pool_allocator
into a unified, production-ready system.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

logger = logging.getLogger(__name__)


class FeatureGate(Enum):
    """Feature gate statuses"""
    OFF = "off"  # Disabled (default)
    EXPERIMENTAL = "experimental"  # Enabled but unstable
    BETA = "beta"  # Mostly stable
    STABLE = "stable"  # Production ready


@dataclass
class FeatureGateConfig:
    """Configuration for a feature gate"""
    name: str
    gate_status: FeatureGate
    min_version: str  # Minimum SGLang/ATOM version required
    supported_hw: List[str]  # e.g., ["gfx1030", "gfx1031"]
    incompatible_with: List[str] = None  # Other features
    
    def __post_init__(self):
        if self.incompatible_with is None:
            self.incompatible_with = []


class TurboQuantFeatureGates:
    """
    Feature gates for TurboQuant KV compression.
    
    Gate Status Summary:
      - 1-bit (TQ1):        EXPERIMENTAL (off by default)
      - 2-bit (TQ2):        BETA (default for gfx1030 after Phase 6 validation)
      - 3-bit (TQ3):        BETA
      - 4-bit (TQ4):        STABLE
      - 8-bit (TQ8):        OFF (reference implementation)
    """
    
    # Feature gate definitions
    gates = {
        "tq1": FeatureGateConfig(
            name="TurboQuant 1-bit",
            gate_status=FeatureGate.EXPERIMENTAL,
            min_version="0.4.0",
            supported_hw=["gfx1030", "gfx1031"],
            incompatible_with=["graph-capture"],  # 1-bit needs careful kernel handling
        ),
        "tq2": FeatureGateConfig(
            name="TurboQuant 2-bit",
            gate_status=FeatureGate.BETA,
            min_version="0.4.0",
            supported_hw=["gfx1030", "gfx1031"],
            incompatible_with=[],
        ),
        "tq3": FeatureGateConfig(
            name="TurboQuant 3-bit",
            gate_status=FeatureGate.BETA,
            min_version="0.4.0",
            supported_hw=["gfx1030", "gfx1031"],
            incompatible_with=[],
        ),
        "tq4": FeatureGateConfig(
            name="TurboQuant 4-bit",
            gate_status=FeatureGate.STABLE,
            min_version="0.4.0",
            supported_hw=["gfx1030", "gfx1031"],
            incompatible_with=[],
        ),
        "tq8": FeatureGateConfig(
            name="TurboQuant 8-bit (reference)",
            gate_status=FeatureGate.OFF,
            min_version="0.4.0",
            supported_hw=["gfx1030", "gfx1031"],
            incompatible_with=[],
        ),
    }
    
    @classmethod
    def is_enabled(cls, mode: str) -> bool:
        """Check if a TurboQuant mode is enabled by feature gate"""
        gate = cls.gates.get(mode)
        if gate is None:
            logger.warning(f"Unknown TurboQuant mode: {mode}")
            return False
        
        if gate.gate_status == FeatureGate.OFF:
            return False
        
        # Check environment override
        env_var = f"SGLANG_ENABLE_{mode.upper()}"
        if env_var in os.environ:
            return os.environ[env_var].lower() in ["true", "1", "yes"]
        
        # EXPERIMENTAL modes: off by default unless explicitly enabled
        if gate.gate_status == FeatureGate.EXPERIMENTAL:
            env_override = os.environ.get("SGLANG_ENABLE_EXPERIMENTAL", "false").lower()
            return env_override in ["true", "1", "yes"]
        
        # BETA/STABLE: on by default
        return True
    
    @classmethod
    def get_default_for_hw(cls, hw: str) -> Optional[str]:
        """
        Get default TurboQuant mode for a hardware platform.
        
        After Phase 6 validation on gfx1030, default will be "tq2".
        Until then, default is "fp16" (no compression).
        """
        if hw == "gfx1030":
            # Post-Phase-6 default will be "tq2"
            # For now, use environment override
            return os.environ.get("SGLANG_DEFAULT_KV_DTYPE", "fp16")
        
        return "fp16"  # Default for unknown hw


class RotorQuantFeatureGates:
    """Feature gates for RotorQuant (future use)"""
    
    gates = {
        "rq3_planar": FeatureGateConfig(
            name="RotorQuant 3-bit Planar",
            gate_status=FeatureGate.EXPERIMENTAL,
            min_version="0.5.0",
            supported_hw=["gfx1030", "gfx1031"],
            incompatible_with=[],
        ),
        "rq4_planar": FeatureGateConfig(
            name="RotorQuant 4-bit Planar",
            gate_status=FeatureGate.EXPERIMENTAL,
            min_version="0.5.0",
            supported_hw=["gfx1030", "gfx1031"],
            incompatible_with=[],
        ),
        "rq3_iso": FeatureGateConfig(
            name="RotorQuant 3-bit Isometric",
            gate_status=FeatureGate.EXPERIMENTAL,
            min_version="0.5.0",
            supported_hw=["gfx1030", "gfx1031"],
            incompatible_with=[],
        ),
        "rq4_iso": FeatureGateConfig(
            name="RotorQuant 4-bit Isometric",
            gate_status=FeatureGate.EXPERIMENTAL,
            min_version="0.5.0",
            supported_hw=["gfx1030", "gfx1031"],
            incompatible_with=[],
        ),
    }
    
    @classmethod
    def is_enabled(cls, mode: str) -> bool:
        """Check if a RotorQuant mode is enabled"""
        gate = cls.gates.get(mode)
        if gate is None:
            return False
        
        # All RotorQuant modes are experimental; require explicit enabling
        env_var = f"SGLANG_ENABLE_{mode.upper()}"
        return os.environ.get(env_var, "false").lower() in ["true", "1", "yes"]


class HardwareSafetyValidator:
    """Validates hardware capabilities and constraints"""
    
    @staticmethod
    def detect_gpu_arch() -> Optional[str]:
        """
        Detect AMD GPU architecture.
        
        Returns:
            "gfx1030", "gfx1031", or None if not AMD or detection fails
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            
            # Check for ROCm
            if hasattr(torch.version, "hip"):
                # Extract GPU architecture from device properties
                props = torch.cuda.get_device_properties(0)
                device_name = props.name if hasattr(props, "name") else ""
                
                if "gfx1030" in device_name:
                    return "gfx1030"
                elif "gfx1031" in device_name:
                    return "gfx1031"
                elif "gfx" in device_name:
                    # Other RDNA2 variant
                    return device_name.split()[-1]
        except Exception as e:
            logger.debug(f"GPU architecture detection failed: {e}")
        
        return None
    
    @staticmethod
    def enforce_gfx1030() -> bool:
        """
        Enforce that we're running on gfx1030.
        
        Can be disabled by setting ENFORCE_GFX1030=false env var.
        
        Returns:
            True if OK to proceed, False if validation failed
        """
        # Check for explicit override
        override = os.environ.get("ENFORCE_GFX1030", "true").lower()
        if override in ["false", "0", "no"]:
            logger.warning("Hardware enforcement disabled via ENFORCE_GFX1030=false")
            return True
        
        arch = HardwareSafetyValidator.detect_gpu_arch()
        
        if arch is None:
            logger.error(
                "Hardware validation failed: AMD GPU not detected\n"
                "  Required: AMD ROCm 7.2+ with gfx1030 GPU\n"
                "  Detected: No AMD GPU found\n"
                "  Workaround: Set --kv-cache-dtype fp16 to disable quantization"
            )
            return False
        
        if arch != "gfx1030":
            logger.warning(
                f"Hardware warning: detected {arch} (not gfx1030)\n"
                f"  TurboQuant optimizations are tuned for gfx1030\n"
                f"  Other RDNA2 variants may work but are untested"
            )
            # Don't fail; allow other RDNA2 variants
        
        return True
    
    @staticmethod
    def validate_compression_config(
        mode: str,
        max_context_length: int = 32000,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate compression configuration for safety.
        
        Args:
            mode: KV cache dtype (e.g., "tq2")
            max_context_length: Max token context
        
        Returns:
            (is_valid, error_message_if_invalid)
        """
        # Check mode is recognized
        valid_modes = [
            "fp16", "fp8_e4m3", "fp8_e5m2", "int8", "int4",
            "tq1", "tq2", "tq3", "tq4", "tq8",
            "rq3_planar", "rq4_planar", "rq3_iso", "rq4_iso"
        ]
        
        if mode not in valid_modes:
            return False, f"Invalid --kv-cache-dtype: {mode}\nValid: {', '.join(valid_modes)}"
        
        # Check feature gate
        if mode.startswith("tq"):
            if not TurboQuantFeatureGates.is_enabled(mode):
                if mode == "tq1":
                    return False, (
                        f"TurboQuant {mode} is EXPERIMENTAL (disabled by default)\n"
                        f"  To enable: export SGLANG_ENABLE_EXPERIMENTAL=true"
                    )
                else:
                    return False, f"TurboQuant {mode} is not enabled"
        
        elif mode.startswith("rq"):
            if not RotorQuantFeatureGates.is_enabled(mode):
                return False, (
                    f"RotorQuant {mode} is EXPERIMENTAL\n"
                    f"  To enable: export SGLANG_ENABLE_{mode.upper()}=true"
                )
        
        # Check context length constraints
        if max_context_length > 32768:
            logger.warning(
                f"Long context requested ({max_context_length})\n"
                f"  Compression may hit memory limits\n"
                f"  Recommended: <= 32000 tokens for gfx1030"
            )
        
        return True, None


class FallbackChainManager:
    """
    Manages graceful fallback when requested mode isn't available.
    
    Fallback strategy:
      tq1 → tq2 → tq3 → tq4 → fp8_e4m3 → fp16
    """
    
    # Default fallback chain for each starting mode
    fallback_chains = {
        "tq1": ["tq2", "tq3", "tq4", "fp8_e4m3", "fp16"],
        "tq2": ["tq3", "tq4", "fp8_e4m3", "fp16"],
        "tq3": ["tq4", "fp8_e4m3", "fp16"],
        "tq4": ["fp8_e4m3", "fp16"],
        "fp8_e4m3": ["fp8_e5m2", "fp16"],
        "fp8_e5m2": ["fp16"],
        "fp16": [],  # No fallback from FP16
    }
    
    @classmethod
    def get_fallback_chain(cls, mode: str) -> List[str]:
        """Get fallback chain for a mode"""
        return cls.fallback_chains.get(mode, ["fp16"])
    
    @classmethod
    def find_available_mode(cls, preferred_mode: str) -> str:
        """
        Find the first available mode in fallback chain.
        
        Args:
            preferred_mode: Requested KV dtype
        
        Returns:
            Available mode (might be fallback)
        """
        # Check if preferred mode is available
        if _is_mode_available(preferred_mode):
            return preferred_mode
        
        # Try fallback chain
        chain = cls.get_fallback_chain(preferred_mode)
        for fallback_mode in chain:
            if _is_mode_available(fallback_mode):
                logger.warning(
                    f"Fallback: {preferred_mode} not available, using {fallback_mode}"
                )
                return fallback_mode
        
        # Shouldn't reach here
        logger.error(f"No available compression mode; using fp16")
        return "fp16"


def _is_mode_available(mode: str) -> bool:
    """Helper: check if a compression mode is available"""
    if mode == "fp16":
        return True  # Always available
    
    if mode.startswith("tq"):
        return TurboQuantFeatureGates.is_enabled(mode)
    
    if mode.startswith("rq"):
        return RotorQuantFeatureGates.is_enabled(mode)
    
    return False


def validate_production_config(
    mode: str,
    enforce_hardware: bool = True,
    allow_experimental: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Comprehensive validation for production deployment.
    
    Args:
        mode: Requested KV cache dtype
        enforce_hardware: Require gfx1030 hardware
        allow_experimental: Allow experimental modes
    
    Returns:
        (is_valid, error_message_if_invalid)
    """
    # Hardware check
    if enforce_hardware:
        if not HardwareSafetyValidator.enforce_gfx1030():
            return False, "Hardware validation failed (see above)"
    
    # Config check
    valid, error = HardwareSafetyValidator.validate_compression_config(mode)
    if not valid:
        return False, error
    
    # Feature gate check
    if not allow_experimental:
        if mode == "tq1" or mode.startswith("rq"):
            # Experimental modes
            return False, (
                f"{mode} is EXPERIMENTAL\n"
                f"  For production: use --kv-cache-dtype tq2 or tq4\n"
                f"  To enable: pass allow_experimental=True"
            )
    
    return True, None


# Module initialization logging
logger.info("Phase 4.5 Feature Gates initialized")
logger.debug(f"  TurboQuant gates: {list(TurboQuantFeatureGates.gates.keys())}")
logger.debug(f"  RotorQuant gates: {list(RotorQuantFeatureGates.gates.keys())}")


from typing import Tuple
