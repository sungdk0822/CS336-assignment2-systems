import os
import timeit
import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.transformer_language_model import TransformerLanguageModel
from cs336_basics.trainer import AdamW, cross_entropy
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from tqdm import tqdm
from typing import Literal


def setup(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def distributed_demo(rank, world_size, data_size, backend):
    setup(rank, world_size, backend)
    device = f'cuda:{rank}' if backend == 'nccl' else 'cpu'
    data = torch.randn(256 * 1000 * data_size, device=device)
    is_nccl = backend == 'nccl'
    warmup_steps = 5
    measurement_steps = 10

    for _ in range(warmup_steps):
        dist.all_reduce(data, async_op=False)

    if is_nccl:
        torch.cuda.synchronize()

    start_time = timeit.default_timer()

    for _ in range(measurement_steps):
        dist.all_reduce(data, async_op=False)

    if is_nccl:
        torch.cuda.synchronize()

    end_time = timeit.default_timer()

    measured_time = (end_time - start_time) / measurement_steps

    if is_nccl:
        measured_time = torch.tensor([measured_time], device=device)
        measured_times = [torch.empty(1, device=device) for _ in range(world_size)]
        dist.all_gather(measured_times, measured_time)
    else:
        measured_times = [None for _ in range(world_size)]
        dist.all_gather_object(measured_times, measured_time)

    if rank == 0:
        avg_time = sum(measured_times) / len(measured_times)
        if is_nccl:
            avg_time = avg_time.item()
        print(f'{avg_time:.4f}')

    dist.destroy_process_group()


def benchmark_all_reduce(
    backend: Literal['gloo', 'nccl'],
    data_size: int, # unit: MB
    num_processes: int
) -> None:
    mp.spawn(fn=distributed_demo, args=(num_processes, data_size, backend), nprocs=num_processes, join=True)


def run_DDP(rank: int, world_size: int, DDP_class: nn.Module) -> None:
    d_model = 1600
    d_ff = 6400
    num_layers = 48
    num_heads = 25
    context_length = 256
    vocab_size = 10000
    batch_size = 16
    # d_model = 16
    # d_ff = 64
    # num_layers = 4
    # num_heads = 2
    # context_length = 256
    # vocab_size = 100
    # batch_size = 16
    local_batch_size = batch_size // world_size
    if torch.cuda.is_available() and world_size <= torch.cuda.device_count():
        backend = 'nccl'
        device = f'cuda:{rank}'
    else:
        backend = 'gloo'
        device = 'cpu'
    if rank == 0:
        print(
            f'gpu count: {torch.cuda.device_count()}', 
            f'world size: {world_size}',
            f'backend: {backend}', 
            f'device: {device}',
            sep='\n'
        )

    setup(rank, world_size, backend=backend)
    torch.manual_seed(rank)

    model = DDP_class(
        TransformerLanguageModel(
                d_model,
                num_heads,
                d_ff,
                10000.0,
                vocab_size,
                context_length,
                num_layers,
                device,
            )
    )
    optimizer = AdamW(model.parameters())
    input_ids = torch.randint(0, vocab_size, (batch_size, context_length))
    label_ids = torch.randint(0, vocab_size, (batch_size, context_length))

    steps = 50
    communication_time_sum = 0
    torch.cuda.synchronize()
    total_time_start = timeit.default_timer()
    step_range = range(steps)
    if rank == 0: 
        step_range = tqdm(step_range)
    for _ in step_range:
        local_input_ids = input_ids[rank * local_batch_size : (rank + 1) * local_batch_size, :].to(device)
        local_label_ids = label_ids[rank * local_batch_size : (rank + 1) * local_batch_size, :].to(device)
        optimizer.zero_grad()
        lm_head_output = model(local_input_ids)
        loss = cross_entropy(lm_head_output, local_label_ids)
        loss.backward()

        torch.cuda.synchronize()
        communication_time_start = timeit.default_timer()
        model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        communication_time_end = timeit.default_timer()
        communication_time_sum += (communication_time_end - communication_time_start)

        optimizer.step()
        permutation = torch.randperm(batch_size)
        input_ids = input_ids[permutation]
        label_ids = label_ids[permutation]
    torch.cuda.synchronize()
    total_time_end = timeit.default_timer()
    total_time_per_step = (total_time_end - total_time_start) / steps
    communication_time_per_step = communication_time_sum / steps
    
    if rank == 0:
        print(f'total time per step: {total_time_per_step:6f}')
        print(f'communication time per step: {communication_time_per_step:6f}')

    dist.destroy_process_group()


def benchmark_DDP(
    DDP_class: nn.Module,
    world_size: int = 2
) -> None:
    mp.spawn(run_DDP, args=(world_size, DDP_class), nprocs=world_size, join=True)
    '''
    results on 4 x A100 single node (unit: s):
        gpu count: 2
        world size: 2
        backend: nccl
        device: cuda

        DDPIndividualParameters
            total time per step: 4.752196
            communication time per step: 0.967147
        DDPMinimalFlat
            total time per step: 4.788055
            communication time per step: 0.939721
    '''


# uv run pytest -k test_DistributedDataParallelIndividualParameters
class DDPIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def finish_gradient_synchronization(self) -> None:
        for parameter in self.module.parameters():
            if parameter.grad is not None:
                # dist.all_reduce(parameter.grad, dist.ReduceOp.AVG) # RuntimeError: Cannot use ReduceOp.AVG with Gloo
                dist.all_reduce(parameter.grad)
                parameter.grad /= dist.get_world_size()


class DDPMinimalFlat(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def finish_gradient_synchronization(self) -> None:
        gradients = [p.grad for p in self.module.parameters() if p.grad is not None]
        flattened_parameters = _flatten_dense_tensors(gradients)
        dist.all_reduce(flattened_parameters)
        gradients = _unflatten_dense_tensors(flattened_parameters, gradients)
        for gradient in gradients:
            gradient /= self.world_size


class DDPOverlapIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        for parameter in self.module.parameters():
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(self.all_reduce_hook)
            dist.broadcast(parameter.data, 0)

    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        return self.module(*inputs, **kwargs)

    # def all_reduce_hook(self, parameter: torch.Tensor) -> None:
    #     dist.all_reduce(parameter.grad, async_op=False)

    # def finish_gradient_synchronization(self) -> None:
    #     for parameter in self.module.parameters():
    #         if parameter.grad is not None:
    #             parameter.grad /= self.world_size

    @nvtx.range('all_reduce_hook')
    def all_reduce_hook(self, parameter: torch.Tensor) -> None:
        parameter.grad /= self.world_size
        dist.all_reduce(parameter.grad, async_op=False)

    @nvtx.range('finish_gradient_synchronization')
    def finish_gradient_synchronization(self) -> None:
        pass


if __name__ == '__main__':
    # benchmark_DDP(DDPIndividualParameters, 4)
    # benchmark_DDP(DDPMinimalFlat, 4)
    benchmark_DDP(DDPOverlapIndividualParameters, 4)
    pass

# uv run nsys profile -o result --force-overwrite true python cs336_systems/distributed_data_parallel.py