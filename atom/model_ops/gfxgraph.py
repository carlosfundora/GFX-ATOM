# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Carlos Fundora / REPLICATOR.
"""gfxGRAPH integration for ATOM — CUDA Graph parity on RDNA2 via Rust bridge.

Provides:
  - BucketRouter: Shape-aware batch bucketing for graph capture reduction.
  - ConditionalGraphRunner: Branch-conditional graph replay with eager fallback.
  - create_atom_bucket_router(): Factory with standard ATOM batch size buckets.

Requirements:
  - gfxgraph_rs PyO3 extension must be built and importable.
  - Enable via: ATOM_ENABLE_GFXGRAPH=1

Usage from model_runner.py or warmup paths:
    from atom.model_ops.gfxgraph import create_atom_bucket_router
    router = create_atom_bucket_router(max_batch=256)
    bucket, state = router.route(current_batch_size)
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("atom")

# Lazy import flag
_gfxgraph_available: Optional[bool] = None
_BucketRouter = None
_ConditionalGraphRunner = None


def _try_import():
    """Attempt to import gfxgraph_rs once; cache result."""
    global _gfxgraph_available, _BucketRouter, _ConditionalGraphRunner
    if _gfxgraph_available is not None:
        return _gfxgraph_available
    try:
        from gfxgraph_rs import BucketRouter, ConditionalGraphRunner
        _BucketRouter = BucketRouter
        _ConditionalGraphRunner = ConditionalGraphRunner
        _gfxgraph_available = True
        logger.info("gfxGRAPH Rust bridge (gfxgraph_rs) loaded successfully")
    except ImportError:
        _gfxgraph_available = False
        logger.warning(
            "gfxGRAPH Rust bridge (gfxgraph_rs) not available. "
            "Install with: cd gfxGRAPH/gfxgraph_rs && maturin develop --release"
        )
    return _gfxgraph_available


def is_enabled() -> bool:
    """Check if gfxGRAPH is both enabled and importable."""
    if os.getenv("ATOM_ENABLE_GFXGRAPH", "0") != "1":
        return False
    return _try_import()


def get_bucket_router():
    """Return the BucketRouter class (or None if unavailable)."""
    if not _try_import():
        return None
    return _BucketRouter


def get_conditional_graph_runner():
    """Return the ConditionalGraphRunner class (or None if unavailable)."""
    if not _try_import():
        return None
    return _ConditionalGraphRunner


# Standard ATOM batch size buckets
_ATOM_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def create_atom_bucket_router(
    max_batch: int = 256,
    extra_buckets: Optional[list[int]] = None,
) -> Optional[object]:
    """Create a BucketRouter pre-configured for ATOM inference.

    Args:
        max_batch: Maximum batch size. Buckets above this are excluded.
        extra_buckets: Additional custom bucket sizes to include.

    Returns:
        BucketRouter instance, or None if gfxGRAPH is unavailable.
    """
    if not _try_import() or _BucketRouter is None:
        return None

    buckets = sorted(set(
        [b for b in _ATOM_BUCKETS if b <= max_batch]
        + (extra_buckets or [])
    ))
    if not buckets:
        buckets = [1]

    router = _BucketRouter(buckets)
    logger.info(
        "gfxGRAPH BucketRouter created with %d buckets: %s",
        len(buckets),
        buckets,
    )
    return router
