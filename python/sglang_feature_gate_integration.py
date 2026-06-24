"""
Phase 4.5.2: SGLang Feature Gate Integration

Bridges sglang_feature_gates with sglang_kv_compression.py to provide:
  - Feature-gate-aware compression backend selection
  - Runtime fallback chain execution
  - Production-safe decoder initialization
  - Configuration validation at startup
  - Telemetry for feature gate usage

This module is called during SGLang server startup to ensure feature gates
are properly enforced before any compression operations begin.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Import feature gates (will fail gracefully if not available)
try:
    from sglang_feature_gates import (
        TurboQuantFeatureGates,
        RotorQuantFeatureGates,
        HardwareSafetyValidator,
        FallbackChainManager,
        validate_production_config,
    )
    FEATURE_GATES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Feature gates not available: {e}")
    FEATURE_GATES_AVAILABLE = False

# Import compression backends
try:
    from sglang_kv_compression import (
        KVCompressionManager,
        get_kv_compression_manager,
    )
    COMPRESSION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"KV compression not available: {e}")
    COMPRESSION_AVAILABLE = False


class SGLangFeatureGateIntegration:
    """
    Integration layer between feature gates and SGLang backends.
    
    Responsibilities:
      1. Validate configuration at startup
      2. Select production-safe backend based on gates
      3. Execute fallback chain on backend errors
      4. Log feature gate usage for observability
    """
    
    def __init__(self):
        self.validation_complete = False
        self.selected_backend = None
        self.fallback_history = []
        self.config = {}
    
    def validate_startup_config(
        self,
        requested_mode: str = "fp16",
        enforce_hardware: bool = True,
        allow_experimental: bool = False,
    ) -> Tuple[bool, Optional[str], str]:
        """
        Validate requested compression mode at startup.
        
        Args:
            requested_mode: KV cache dtype (default: fp16)
            enforce_hardware: Require gfx1030 hardware
            allow_experimental: Allow experimental modes
        
        Returns:
            (is_valid, error_msg, effective_mode)
        """
        if not FEATURE_GATES_AVAILABLE:
            logger.warning("Feature gates not available; proceeding with fp16")
            return True, None, "fp16"
        
        # Validate production config
        is_valid, error = validate_production_config(
            requested_mode,
            enforce_hardware=enforce_hardware,
            allow_experimental=allow_experimental
        )
        
        if not is_valid:
            if not allow_experimental and error and "EXPERIMENTAL" in error:
                # Try to find a stable fallback
                logger.warning(f"Requested mode is experimental; falling back")
                fallback = FallbackChainManager.find_available_mode(requested_mode)
                logger.info(f"Using fallback mode: {fallback}")
                self.fallback_history.append((requested_mode, fallback))
                self.validation_complete = True
                self.selected_backend = fallback
                self.config = {
                    "requested": requested_mode,
                    "effective": fallback,
                    "fallback_reason": "experimental_disabled",
                }
                return True, None, fallback
            else:
                # Fatal error
                logger.error(f"Configuration validation failed: {error}")
                return False, error, "fp16"
        
        self.validation_complete = True
        self.selected_backend = requested_mode
        self.config = {
            "requested": requested_mode,
            "effective": requested_mode,
            "fallback_reason": None,
        }
        return True, None, requested_mode
    
    def get_compression_backend(self, mode: str) -> Optional[Dict[str, Any]]:
        """
        Get compression backend configuration for a mode.
        
        Returns None if mode unavailable; caller should fallback to fp16.
        """
        if not FEATURE_GATES_AVAILABLE:
            return None
        
        # Check if feature gate is enabled
        if mode.startswith("tq"):
            if not TurboQuantFeatureGates.is_enabled(mode):
                logger.debug(f"TurboQuant mode {mode} not enabled by feature gate")
                return None
        elif mode.startswith("rq"):
            if not RotorQuantFeatureGates.is_enabled(mode):
                logger.debug(f"RotorQuant mode {mode} not enabled by feature gate")
                return None
        
        # Return backend config
        return {
            "mode": mode,
            "gate_enabled": True,
            "compression_type": "turboquant" if mode.startswith("tq") else "rotorquant",
        }
    
    def execute_fallback_chain(self, preferred_mode: str) -> str:
        """
        Execute fallback chain: try preferred, then fallback options.
        
        Args:
            preferred_mode: Requested mode (e.g., "tq2")
        
        Returns:
            Available mode name
        """
        if not FEATURE_GATES_AVAILABLE:
            return "fp16"
        
        return FallbackChainManager.find_available_mode(preferred_mode)
    
    def log_startup_summary(self):
        """Log feature gate configuration summary at startup"""
        if not self.validation_complete:
            logger.warning("Feature gate validation not completed")
            return
        
        logger.info("=" * 70)
        logger.info("SGLang Feature Gate Configuration Summary")
        logger.info("=" * 70)
        
        if self.config:
            logger.info(f"  Requested Mode:    {self.config.get('requested', 'N/A')}")
            logger.info(f"  Effective Mode:    {self.config.get('effective', 'N/A')}")
            if self.config.get('fallback_reason'):
                logger.info(f"  Fallback Reason:   {self.config['fallback_reason']}")
        
        if self.fallback_history:
            logger.info(f"  Fallback Chain:    {' → '.join(chain[1] for chain in self.fallback_history)}")
        
        # Log enabled feature gates
        if FEATURE_GATES_AVAILABLE:
            enabled_modes = []
            for mode in ["tq1", "tq2", "tq3", "tq4"]:
                if TurboQuantFeatureGates.is_enabled(mode):
                    enabled_modes.append(f"{mode}✓")
            if enabled_modes:
                logger.info(f"  Enabled Modes:     {', '.join(enabled_modes)}")
        
        logger.info("=" * 70)


class SGLangCompressionPipelineWithGates:
    """
    Enhanced compression pipeline that respects feature gates.
    
    This wrapper ensures that SGLang's compression backends are gated
    and can fail over gracefully.
    """
    
    def __init__(self, manager: Optional[KVCompressionManager] = None):
        """
        Initialize compression pipeline with feature gate awareness.
        
        Args:
            manager: KVCompressionManager instance (get from get_kv_compression_manager if None)
        """
        self.manager = manager if manager else (get_kv_compression_manager() if COMPRESSION_AVAILABLE else None)
        self.integration = SGLangFeatureGateIntegration()
        self.backend_errors = {}  # Track which backends have failed
    
    def initialize_with_config(
        self,
        config: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Initialize compression pipeline from SGLang config.
        
        Args:
            config: SGLang config dict with "kv_cache_dtype" key
        
        Returns:
            (success, error_msg)
        """
        mode = config.get("kv_cache_dtype", "fp16")
        enforce_hw = config.get("enforce_gfx1030", True)
        allow_exp = config.get("allow_experimental_modes", False)
        
        # Validate startup config
        is_valid, error, effective_mode = self.integration.validate_startup_config(
            mode,
            enforce_hardware=enforce_hw,
            allow_experimental=allow_exp
        )
        
        if not is_valid:
            return False, error
        
        logger.info(f"KV compression: mode={effective_mode} (requested={mode})")
        
        # Initialize compression manager with effective mode
        if self.manager and COMPRESSION_AVAILABLE:
            try:
                # Get compression backend
                backend_config = self.integration.get_compression_backend(effective_mode)
                if backend_config:
                    logger.debug(f"Using compression backend: {backend_config}")
            except Exception as e:
                logger.error(f"Failed to initialize compression backend: {e}")
                return False, str(e)
        
        # Log startup summary
        self.integration.log_startup_summary()
        
        return True, None
    
    def encode_kv_with_fallback(self, kv_data, mode: str, layer_idx: int = 0):
        """
        Encode KV with automatic fallback on failure.
        
        Args:
            kv_data: KV tensor to encode
            mode: Compression mode
            layer_idx: Layer index for logging
        
        Returns:
            Encoded KV data, or falls back to FP16 version on error
        """
        if not self.manager:
            return kv_data  # No compression available
        
        try:
            # Try primary mode
            result = self.manager.encode(kv_data, mode)
            return result
        except Exception as e:
            logger.warning(f"Encode failed for {mode} at layer {layer_idx}: {e}")
            self.backend_errors[mode] = str(e)
            
            # Try fallback chain
            fallback_mode = self.integration.execute_fallback_chain(mode)
            if fallback_mode != mode:
                logger.info(f"Falling back to {fallback_mode}")
                try:
                    result = self.manager.encode(kv_data, fallback_mode)
                    return result
                except Exception as e2:
                    logger.error(f"Fallback encode also failed: {e2}")
                    return kv_data
            
            return kv_data
    
    def decode_kv_with_fallback(self, compressed_kv, mode: str, layer_idx: int = 0):
        """
        Decode compressed KV with automatic fallback on failure.
        
        Args:
            compressed_kv: Compressed KV data
            mode: Compression mode
            layer_idx: Layer index for logging
        
        Returns:
            Decoded KV, or original if fallback fails
        """
        if not self.manager or not hasattr(compressed_kv, 'data'):
            return compressed_kv
        
        try:
            result = self.manager.decode(compressed_kv)
            return result
        except Exception as e:
            logger.warning(f"Decode failed for {mode} at layer {layer_idx}: {e}")
            
            # Try fallback (less critical than encode, just return as-is)
            logger.error(f"Fallback decode not implemented; returning compressed data")
            self.backend_errors[mode] = str(e)
            return compressed_kv


# Global integration instance
_global_integration: Optional[SGLangFeatureGateIntegration] = None


def get_feature_gate_integration() -> SGLangFeatureGateIntegration:
    """Get or create global feature gate integration instance"""
    global _global_integration
    if _global_integration is None:
        _global_integration = SGLangFeatureGateIntegration()
    return _global_integration


def init_compression_with_gates(
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Initialize compression pipeline with feature gates.
    
    Args:
        config: SGLang config dict (uses defaults if None)
    
    Returns:
        (success, error_msg)
    """
    if config is None:
        config = {}
    
    integration = get_feature_gate_integration()
    mode = config.get("kv_cache_dtype", "fp16")
    enforce_hw = config.get("enforce_gfx1030", True)
    allow_exp = config.get("allow_experimental_modes", False)
    
    is_valid, error, effective_mode = integration.validate_startup_config(
        mode,
        enforce_hardware=enforce_hw,
        allow_experimental=allow_exp
    )
    
    if not is_valid:
        logger.error(f"Initialization failed: {error}")
        return False, error
    
    logger.info(f"Compression initialized with mode: {effective_mode}")
    integration.log_startup_summary()
    return True, None


logger.info("Phase 4.5.2 SGLang Feature Gate Integration loaded")
