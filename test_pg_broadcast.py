import torch
import torch.distributed as dist

print(dist.ProcessGroup.broadcast.__doc__)
