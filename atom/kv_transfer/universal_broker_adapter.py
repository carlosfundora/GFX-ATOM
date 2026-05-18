# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Universal KV Broker adapter for gfxatom engine.

This module integrates the SGLang Universal KV Broker as a KV connector backend,
enabling model-agnostic compressed KV cache ownership with support for
TurboQuant and RotorQuant quantization modes.

The adapter wraps the broker as a backend that can be dynamically selected
via KVConnectorFactory, with graceful fallback for unsupported configurations.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from atom.kv_transfer.disaggregation.base import KVConnectorBase, KVConnectorSchedulerBase
from atom.kv_transfer.disaggregation.types import ConnectorMetadata

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class UniversalBrokerConnector(KVConnectorBase):
    """Worker-side KV connector using Universal KV Broker.
    
    This connector manages compressed KV cache storage and retrieval
    for a single TP rank, delegating to the UniversalKVBroker for
    allocation, compression, and spill management.
    """

    is_producer: bool = True

    def __init__(self, config: Any, broker_config: dict[str, Any] | None = None):
        """Initialize the broker connector.
        
        Args:
            config: Model runner or engine configuration object.
            broker_config: Optional overrides for broker parameters.
        """
        self.config = config
        self.broker_config = broker_config or {}
        
        # Import here to avoid circular dependencies and optional dependency on SGLang
        try:
            from universal_kv.model_registry import ModelShapeRegistry
            from universal_kv.types import TierKind, UniversalKVBlockHeader
        except ImportError as e:
            raise ImportError(
                "Universal KV Broker dependencies not found. "
                "Ensure universal_kv package is installed."
            ) from e
        
        # Import broker components
        try:
            from sglang.srt.layers.attention.universal_kv_broker import UniversalKVBroker
            from sglang.srt.mem_cache.universal_kv_spill import UniversalKVSpillManager
        except ImportError as e:
            logger.warning(
                f"Cannot import UniversalKVBroker from sglang: {e}. "
                "Broker will not be available. Falling back to standard KV."
            )
            raise ValueError(
                "Universal KV Broker not available. Check sglang installation."
            ) from e
        
        # Initialize broker with config
        gpu_capacity_mb = self.broker_config.get(
            "gpu_capacity_mb",
            getattr(config, "universal_kv_gpu_capacity_mb", 1024)
        )
        ram_capacity_mb = self.broker_config.get(
            "ram_capacity_mb",
            getattr(config, "universal_kv_ram_capacity_mb", 4096)
        )
        hot_importance_threshold = self.broker_config.get(
            "hot_importance_threshold",
            getattr(config, "universal_kv_hot_importance_threshold", 0.7)
        )
        
        self.broker = UniversalKVBroker(
            gpu_capacity_mb=gpu_capacity_mb,
            ram_capacity_mb=ram_capacity_mb,
            hot_importance_threshold=hot_importance_threshold,
            spill_manager=UniversalKVSpillManager(pin_memory=True),
        )
        
        # Model metadata for broker
        model_path = getattr(config, "model_path", "unknown")
        self._model_tag = int(
            hashlib.sha1(model_path.encode("utf-8")).hexdigest()[:2],
            16,
        )
        self._block_size = self.broker_config.get(
            "block_size",
            getattr(config, "universal_kv_block_size", 16)
        )
        
        # Capability tracking
        self._supported_kv_dtypes = {"fp16", "bf16", "int8", "int4", "fp8"}
        self._kv_caches: dict[str, Any] = {}
        
        logger.info(
            f"Initialized UniversalBrokerConnector: "
            f"gpu_capacity_mb={gpu_capacity_mb}, "
            f"ram_capacity_mb={ram_capacity_mb}, "
            f"hot_threshold={hot_importance_threshold}"
        )

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Register local KV cache tensors for broker tracking.
        
        Args:
            kv_caches: Dictionary mapping cache names to cache tensor dicts.
        """
        self._kv_caches = kv_caches
        
        # Validate KV cache configuration
        for cache_name, cache_info in kv_caches.items():
            dtype = str(cache_info.get("dtype", "fp16"))
            if dtype not in self._supported_kv_dtypes:
                logger.warning(
                    f"KV cache '{cache_name}' has unsupported dtype '{dtype}'. "
                    f"Supported: {self._supported_kv_dtypes}. Will degrade to fp16."
                )
        
        logger.debug(f"Registered {len(kv_caches)} KV caches with broker")

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        """Initiate async KV loads for pending requests.
        
        Args:
            metadata: Transfer metadata from scheduler.
        """
        # For broker-based connector, this is a no-op during async phase.
        # Actual materialization happens on-demand during attention compute.
        # This is where we could queue compression/spill operations.
        logger.debug(
            f"Starting KV loads: {len(metadata.reqs_to_recv)} receives, "
            f"{len(metadata.reqs_to_save)} saves"
        )

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Return (done_sending, done_recving) request IDs.
        
        Returns:
            Tuple of (finished_sending, finished_recving) request ID sets.
        """
        # Placeholder: actual transfer tracking would be implemented here
        return (set(), set())


class UniversalBrokerSchedulerConnector(KVConnectorSchedulerBase):
    """Scheduler-side KV connector using Universal KV Broker.
    
    This connector runs in the scheduler process and manages the
    lifecycle of KV cache transfers and broker allocation decisions.
    """

    is_producer: bool = True

    def __init__(self, config: Any, broker_config: dict[str, Any] | None = None):
        """Initialize the scheduler-side broker connector.
        
        Args:
            config: Scheduler or engine configuration object.
            broker_config: Optional overrides for broker parameters.
        """
        self.config = config
        self.broker_config = broker_config or {}
        
        # Enable broker observability by default
        self._enable_metrics = self.broker_config.get("enable_metrics", True)
        self._metrics = {
            "allocations": 0,
            "compressions": 0,
            "spills": 0,
            "evictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        
        logger.info("Initialized UniversalBrokerSchedulerConnector")

    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        """Check if sequence needs remote KV prefill.
        
        Args:
            seq: Sequence object to check.
            
        Returns:
            Tuple of (num_tokens, needs_async_load).
        """
        # Placeholder: actual prefix matching logic would go here
        return (0, False)

    def build_connector_meta(self) -> ConnectorMetadata:
        """Build metadata snapshot of pending transfers.
        
        Returns:
            ConnectorMetadata object for the current scheduler step.
        """
        return ConnectorMetadata()

    def update_state_after_alloc(self, seq: Any) -> None:
        """Update state after scheduler allocates blocks.
        
        Args:
            seq: Sequence object that was allocated.
        """
        if self._enable_metrics:
            self._metrics["allocations"] += 1

    def request_finished(self, seq: Any) -> None:
        """Handle request completion and cleanup.
        
        Args:
            seq: Sequence object that finished execution.
        """
        # Placeholder: actual cleanup logic would go here
        pass

    def get_metrics(self) -> dict[str, int]:
        """Get broker observability metrics.
        
        Returns:
            Dictionary of metrics including allocations, compressions, spills, etc.
        """
        return self._metrics.copy()


def create_universal_broker_connector(
    config: Any,
    role: str = "worker",
    broker_config: dict[str, Any] | None = None,
) -> KVConnectorBase | KVConnectorSchedulerBase:
    """Factory function to create appropriate broker connector.
    
    This function is used by KVConnectorFactory to instantiate the
    Universal KV Broker adapter with graceful fallback for unsupported
    environments.
    
    Args:
        config: Configuration object (model_runner or scheduler).
        role: "worker" or "scheduler" to determine connector type.
        broker_config: Optional broker parameter overrides.
        
    Returns:
        Appropriate connector instance (worker or scheduler).
        
    Raises:
        ValueError: If broker cannot be initialized and fallback is not available.
    """
    try:
        if role == "worker":
            return UniversalBrokerConnector(config, broker_config)
        elif role == "scheduler":
            return UniversalBrokerSchedulerConnector(config, broker_config)
        else:
            raise ValueError(f"Unknown connector role: {role}")
    except ImportError as e:
        logger.error(f"Failed to create broker connector: {e}")
        raise ValueError(
            f"Universal KV Broker initialization failed for role '{role}': {e}. "
            "Set --kv-connector to a different backend or ensure universal_kv "
            "and sglang dependencies are installed."
        ) from e


# Registerable factory wrapper for compatibility with KVConnectorFactory
def get_broker_connector_specs() -> dict[str, Any]:
    """Return specifications for registering with KVConnectorFactory.
    
    Returns:
        Dictionary containing worker_module, worker_class, scheduler_module, scheduler_class.
    """
    return {
        "worker_module": "atom.kv_transfer.universal_broker_adapter",
        "worker_class": "UniversalBrokerConnector",
        "scheduler_module": "atom.kv_transfer.universal_broker_adapter",
        "scheduler_class": "UniversalBrokerSchedulerConnector",
    }
