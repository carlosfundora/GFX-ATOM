from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BackendDecision:
    backend: str
    reason: str


class KernelBackend(Protocol):
    name: str

    def supports(self, op_name: str, **kwargs) -> bool: ...
