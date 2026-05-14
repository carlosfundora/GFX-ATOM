import torch
import torch.distributed as dist
import multiprocessing as mp
import socket
import ipaddress
from torch.distributed.distributed_c10d import Backend, PrefixStore, _get_default_timeout
from torch.distributed.rendezvous import rendezvous
from datetime import timedelta

def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False

def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"

def init_gloo_process_group(
    backend: Backend,
    prefix_store: PrefixStore,
    group_rank: int,
    group_size: int,
    timeout: timedelta,
) -> dist.ProcessGroup:
    pg = dist.ProcessGroup(prefix_store, group_rank, group_size)
    from torch.distributed.distributed_c10d import ProcessGroupGloo
    backend_class = ProcessGroupGloo(prefix_store, group_rank, group_size, timeout=timeout)
    backend_type = dist.ProcessGroup.BackendType.GLOO
    device = torch.device("cpu")
    pg._set_default_backend(backend_type)
    backend_class._set_sequence_number_for_group()
    pg._register_backend(device, backend_type, backend_class)
    return pg

def stateless_init_torch_distributed_process_group(host, port, rank, world_size, backend):
    init_method = get_tcp_uri(host, port)
    backend = Backend(backend)
    timeout = _get_default_timeout(backend)
    store, rank, world_size = next(rendezvous(init_method, rank, world_size, timeout=timeout))
    store.set_timeout(timeout)
    prefix_store = PrefixStore(init_method, store)
    return init_gloo_process_group(backend, prefix_store, rank, world_size, timeout)

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
    try:
        dist.broadcast(t, group=pg, group_src=1)
        print(f"Rank {rank} dist.broadcast result: {t.item()}")
    except Exception as e:
        print(f"Rank {rank} dist.broadcast failed: {e}")

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
