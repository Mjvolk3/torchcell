# minimal_ddp_test.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def ddp_test(rank, world_size):
    setup(rank, world_size)

    device = torch.device(f'cuda:{rank}')

    model = nn.Linear(10, 10).to(device)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10).to(device))
    labels = torch.randn(20, 10).to(device)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"Rank {rank} finished with loss {loss.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(ddp_test, args=(world_size,), nprocs=world_size)
