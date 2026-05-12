from __future__ import annotations

from atom.kernel_backend.interface import BackendDecision
from atom.kernel_backend.selector import BackendSelector
from atom.utils import envs


def select_kernel_backend(op_name: str, te_supported: bool) -> tuple[str, str]:
    selector = BackendSelector(
        mode=envs.ATOM_KERNEL_BACKEND_MODE,
        te_allowlist=envs.ATOM_TE_OP_ALLOWLIST,
        te_denylist=envs.ATOM_TE_OP_DENYLIST,
    )
    decision: BackendDecision = selector.select_backend(op_name, te_supported)
    return decision.backend, decision.reason


__all__ = ["BackendDecision", "BackendSelector", "select_kernel_backend"]
