from pathlib import Path

import pytest

from atom.kernel_backend.selector import BackendSelector


def test_layernorm_wires_selector_hook():
    layernorm_file = (
        Path(__file__).resolve().parents[1] / "atom" / "model_ops" / "layernorm.py"
    )
    src = layernorm_file.read_text(encoding="utf-8")
    assert 'op_name="rmsnorm"' in src
    assert "select_kernel_backend(" in src


def test_rmsnorm_selector_uses_aiter_in_forced_aiter_mode():
    selector = BackendSelector(mode="aiter", te_allowlist="", te_denylist="")
    decision = selector.select_backend(op_name="rmsnorm", te_supported=False)
    assert decision.backend == "aiter"
    assert "forced backend 'aiter'" == decision.reason


def test_rmsnorm_selector_fails_loud_in_forced_te_mode():
    selector = BackendSelector(mode="te", te_allowlist="", te_denylist="")
    with pytest.raises(RuntimeError, match="forced backend 'te'"):
        selector.select_backend(op_name="rmsnorm", te_supported=False)
