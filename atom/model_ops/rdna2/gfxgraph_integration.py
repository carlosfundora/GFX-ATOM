# SPDX-License-Identifier: Apache-2.0
"""gfxGRAPH integration for ATOM — CUDA Graph parity on AMD RDNA2 (gfx1030).

Activates gfxGRAPH's transparent monkey-patching of torch.cuda.CUDAGraph
when running on gfx1030 hardware. This bridges the 4 CUDA Graph parity gaps:

  1. Dynamic shape support via ShapeBucketPool
  2. Conditional graph execution via per-branch dispatch
  3. Capture compositor for complex graph topologies
  4. Shape manager for runtime shape selection

Usage in ATOM:
    from atom.model_ops.rdna2.gfxgraph_integration import maybe_enable_gfxgraph
    maybe_enable_gfxgraph()  # auto-detects gfx1030 and enables if available

Or via env var:
    ATOM_GFXGRAPH=1 python -m atom.entrypoints.openai_server ...
"""

import logging
import os

logger = logging.getLogger("atom")

_gfxgraph_enabled = False


def is_rdna2() -> bool:
    """Check if the current GPU is gfx1030 (RDNA2)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        return "gfx1030" in gcn_arch
    except Exception:
        return False


def maybe_enable_gfxgraph(
    *,
    force: bool = False,
    validate: bool = False,
    debug: bool = False,
) -> bool:
    """Enable gfxGRAPH if on RDNA2 hardware and the package is available.

    Args:
        force: Enable even if not on gfx1030 (for testing).
        validate: Enable validation mode (compare graph vs eager output).
        debug: Enable verbose debug logging.

    Returns:
        True if gfxGRAPH was enabled, False otherwise.
    """
    global _gfxgraph_enabled

    if _gfxgraph_enabled:
        return True

    # Check env var override
    env_val = os.environ.get("ATOM_GFXGRAPH", "").lower()
    if env_val == "0" or env_val == "off":
        logger.debug("gfxGRAPH disabled via ATOM_GFXGRAPH=0")
        return False
    if env_val == "1" or env_val == "on":
        force = True
    if env_val == "validate":
        force = True
        validate = True
    if env_val == "debug":
        force = True
        debug = True

    # Auto-detect RDNA2
    if not force and not is_rdna2():
        logger.debug("gfxGRAPH skipped — not on gfx1030 hardware")
        return False

    try:
        import gfxgraph
        gfxgraph.enable(validate=validate, debug=debug)
        _gfxgraph_enabled = True
        logger.info(
            "gfxGRAPH v%s enabled for CUDA Graph parity on RDNA2",
            getattr(gfxgraph, "__version__", "?"),
        )
        return True
    except ImportError:
        logger.info(
            "gfxGRAPH not installed — CUDA Graphs may have limited "
            "functionality on gfx1030. Install with: uv pip install gfxgraph"
        )
        return False
    except Exception as e:
        logger.warning("gfxGRAPH enable failed: %s", e)
        return False


def gfxgraph_stats() -> dict | None:
    """Return gfxGRAPH performance counters, or None if not enabled."""
    if not _gfxgraph_enabled:
        return None
    try:
        import gfxgraph
        return gfxgraph.stats()
    except Exception:
        return None


def gfxgraph_health() -> dict | None:
    """Run gfxGRAPH health check, or None if not available."""
    try:
        import gfxgraph
        return gfxgraph.health_check()
    except ImportError:
        return None
    except Exception as e:
        return {"ok": False, "details": str(e)}
