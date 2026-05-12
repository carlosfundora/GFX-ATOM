import torch
import torch.distributed as dist
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"

try:
    dist.init_process_group("gloo", rank=0, world_size=1)

    t = torch.tensor([1, 2, 3])

    # Try broadcast
    try:
        dist.broadcast(t, src=0)
        print("broadcast available directly on dist!")
    except Exception as e:
        print(f"dist.broadcast error: {e}")

    try:
        group = dist.group.WORLD
        # The docs say `broadcast` is on dist, but we might want to check the ProcessGroup instance
        print("ProcessGroup WORLD type:", type(group))

        # Does the object returned by our function support broadcast?
        print("Does ProcessGroup have broadcast attribute?", hasattr(dist.ProcessGroup, "broadcast"))

    except Exception as e:
        print(f"Error checking ProcessGroup: {e}")

    dist.destroy_process_group()
except Exception as e:
    print(f"Initialization error: {e}")
