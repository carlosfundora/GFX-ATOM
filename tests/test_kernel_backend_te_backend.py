from atom.kernel_backend.backends.te_backend import TransformerEngineBackend
from atom.kernel_backend.selector import BackendSelector


def test_te_backend_supports_reports_false_when_te_missing(monkeypatch):
    monkeypatch.setattr(
        "atom.kernel_backend.backends.te_backend._import_te", lambda: None
    )
    te = TransformerEngineBackend()
    assert te.supports("rmsnorm") is False


def test_forced_te_mode_raises_loud_error_when_missing():
    sel = BackendSelector(mode="te", te_allowlist="", te_denylist="")
    try:
        sel.select_backend(op_name="rmsnorm", te_supported=False)
        assert False, "Expected RuntimeError for forced TE mode"
    except RuntimeError as exc:
        assert "forced backend 'te'" in str(exc)
