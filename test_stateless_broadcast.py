import torch
import torch.distributed as dist
import multiprocessing as mp
from atom.utils.distributed.utils import stateless_init_torch_distributed_process_group
import socket

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def worker(rank, world_size, port):
    pg = stateless_init_torch_distributed_process_group(
        host="127.0.0.1", port=port, rank=rank, world_size=world_size, backend="gloo"
    )

    t = torch.tensor([rank], dtype=torch.int32)

    # Try allreduce
    pg.allreduce([t])
    print(f"Rank {rank} allreduce result: {t.item()}")

    t2 = torch.tensor([rank], dtype=torch.int32)
    # Try broadcast (from rank 1)
    opts = dist.BroadcastOptions()
    opts.rootRank = 1
    # Try using ProcessGroup.broadcast with options
    try:
        pg.broadcast([t2], opts).wait()
        print(f"Rank {rank} broadcast (opts) result: {t2.item()}")
    except Exception as e:
        print(f"Rank {rank} broadcast (opts) failed: {e}")

    t3 = torch.tensor([rank], dtype=torch.int32)
    # Try using ProcessGroup.broadcast without options
    try:
        pg.broadcast(t3, root=1).wait()
        print(f"Rank {rank} broadcast (root) result: {t3.item()}")
    except Exception as e:
        print(f"Rank {rank} broadcast (root) failed: {e}")

if __name__ == "__main__":
    world_size = 2
    port = get_free_port()
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
