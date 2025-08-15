import os
import timeit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Literal


def setup(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def distributed_demo(rank, world_size, data_size, backend):
    setup(rank, world_size, backend)
    device = f'cuda:{rank}' if backend == 'nccl' else 'cpu'
    data = torch.randn(256 * 1000 * data_size, device=device)

    if backend == 'nccl':
        torch.cuda.synchronize()

    start_time = timeit.default_timer()

    dist.all_reduce(data, async_op=False)

    if backend == 'nccl':
        torch.cuda.synchronize()

    end_time = timeit.default_timer()

    measured_time = end_time - start_time

    measured_times = [None for _ in range(world_size)]

    dist.all_gather_object(measured_times, measured_time)

    if rank == 0:
        avg_time = sum(measured_times) / len(measured_times)
        print(f'{avg_time:.4f}')


def benchmark_all_reduce(
    backend: Literal['gloo', 'nccl'],
    data_size: int, # unit: MB
    num_processes: int
) -> None:
    mp.spawn(fn=distributed_demo, args=(num_processes, data_size, backend), nprocs=num_processes, join=True)


if __name__ == '__main__':
    benchmark_all_reduce('gloo', 1000, 6)