import pytest

from atom.kernel_backend.selector import BackendSelector


def test_selector_forced_te_raises_when_te_unsupported():
    sel = BackendSelector(mode="te", te_allowlist="", te_denylist="")
    with pytest.raises(RuntimeError, match="forced backend 'te'"):
        sel.select_backend(op_name="rmsnorm", te_supported=False)


def test_selector_auto_falls_back_to_aiter():
    sel = BackendSelector(mode="auto", te_allowlist="", te_denylist="")
    decision = sel.select_backend(op_name="rmsnorm", te_supported=False)
    assert decision.backend == "aiter"
    assert "unsupported" in decision.reason


def test_selector_allowlist_respected():
    sel = BackendSelector(
        mode="auto", te_allowlist="rmsnorm,linear", te_denylist=""
    )
    decision = sel.select_backend(op_name="attention_mha", te_supported=True)
    assert decision.backend == "aiter"


def test_selector_auto_prefers_te_when_supported():
    sel = BackendSelector(mode="auto", te_allowlist="", te_denylist="")
    decision = sel.select_backend(op_name="rmsnorm", te_supported=True)
    assert decision.backend == "te"
    assert decision.reason == "auto selected te"
