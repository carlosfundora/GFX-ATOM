class AiterBackend:
    name = "aiter"

    def supports(self, op_name: str, **kwargs) -> bool:
        return True
