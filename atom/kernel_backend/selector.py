from __future__ import annotations

from atom.kernel_backend.interface import BackendDecision


def _csv_to_set(value: str) -> set[str]:
    return {x.strip() for x in value.split(",") if x.strip()}


class BackendSelector:
    def __init__(self, mode: str, te_allowlist: str, te_denylist: str):
        self.mode = mode
        self.allow = _csv_to_set(te_allowlist)
        self.deny = _csv_to_set(te_denylist)

    def select_backend(self, op_name: str, te_supported: bool) -> BackendDecision:
        if op_name in self.deny:
            return BackendDecision("aiter", "denied by ATOM_TE_OP_DENYLIST")
        if self.allow and op_name not in self.allow:
            return BackendDecision("aiter", "not in ATOM_TE_OP_ALLOWLIST")
        if self.mode == "aiter":
            return BackendDecision("aiter", "forced backend 'aiter'")
        if self.mode == "te":
            if not te_supported:
                raise RuntimeError(
                    f"forced backend 'te' does not support op '{op_name}'"
                )
            return BackendDecision("te", "forced backend 'te'")
        if te_supported:
            return BackendDecision("te", "auto selected te")
        return BackendDecision("aiter", "te unsupported for op")
