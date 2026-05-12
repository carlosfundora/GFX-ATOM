import torch
import torch.distributed as dist

opts = dist.BroadcastOptions()
print(dir(opts))
print(opts.rootRank)
print(opts.rootTensor)
