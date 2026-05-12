import re

content = """
    A replacement for `torch.distributed.init_process_group` that does not
    pollute the global state. The created ProcessGroup object can be used for
    some operations such as `allreduce`, because it does not depend on the
    global rank. However, some operations such as `broadcast` cannot be used
    because it depends on the global rank.

    # TODO: ask for help from PyTorch team if we need the `broadcast` operation.
"""

new_content = """
    A replacement for `torch.distributed.init_process_group` that does not
    pollute the global state. The created ProcessGroup object can be used for
    collective operations like `allreduce` and `broadcast`. For `broadcast`,
    since the process group is not registered globally, you must use the `group_src`
    argument instead of `src` (e.g., `dist.broadcast(tensor, group=pg, group_src=rank)`).
"""
