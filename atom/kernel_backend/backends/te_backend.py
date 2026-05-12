from __future__ import annotations


def _import_te():
    try:
        import transformer_engine.pytorch as te

        return te
    except Exception:
        return None


class TransformerEngineBackend:
    name = "te"

    def __init__(self):
        self.te = _import_te()

    def supports(self, op_name: str, **kwargs) -> bool:
        if self.te is None:
            return False
        return op_name in {"rmsnorm", "linear", "attention_mha"}
