import timeit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.transformer_language_model import TransformerLanguageModel
from cs336_basics.trainer import cross_entropy
from cs336_systems.distributed_data_parallel import setup
from torch.optim import Optimizer, AdamW
from tqdm import tqdm
from typing import Any


# uv run pytest tests/test_sharded_optimizer.py
class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Optimizer, **kwargs: Any):
        '''
        Initializes the
        sharded state optimizer. params is a collection of parameters to be optimized (or parameter
        groups, in case the user wants to use different hyperparameters, such as learning rates, for differ-
        ent parts of the model); these parameters will be sharded across all the ranks. The optimizer_cls
        parameter specifies the type of optimizer to be wrapped (e.g., optim.AdamW). Finally, any remain-
        ing keyword arguments are forwarded to the constructor of the optimizer_cls. Make sure to
        call the torch.optim.Optimizer super-class constructor in this method.
        '''
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.params = []
        self.assigned_params_per_rank = []
        super().__init__(params, kwargs)
        self.optimizer = optimizer_cls(self.params, **kwargs)

    def step(self, closure = None, **kwargs):
        '''
        Calls the wrapped optimizer's step() method with the pro-
        vided closure and keyword arguments. After updating the parameters, synchronize with the other
        ranks.
        '''
        self.optimizer.step(closure, **kwargs)
        with torch.no_grad():
            for rank in range(self.world_size):
                for param in self.assigned_params_per_rank[rank]:
                    dist.broadcast(param.data, rank)

    def add_param_group(self, param_group: dict[str, Any]): 
        '''
        This method should add a parame-
        ter group to the sharded optimizer. This is called during construction of the sharded optimizer by
        the super-class constructor and may also be called during training (e.g., for gradually unfreezing
        layers in a model). As a result, this method should handle assigning the model's parameters
        among the ranks.
        '''
        all_params = []
        for group in param_group.items():
            for param in group[-1]:
                all_params.append(param)

        assigned_bytes_per_rank = [(rank, 0) for rank in range(self.world_size)]
        assigned_params_per_rank = [[] for _ in range(self.world_size)]
        all_params.sort(key=lambda p: p.nbytes)

        i = 0
        while len(all_params) > 0:
            if i == 0:
                assigned_bytes_per_rank.sort(key=lambda tuple: tuple[1])
            param = all_params.pop()
            rank = assigned_bytes_per_rank[i][0]
            assigned_bytes = assigned_bytes_per_rank[i][1]
            assigned_bytes_per_rank[i] = (rank, assigned_bytes + param.nbytes)
            assigned_params_per_rank[rank].append(param)
            i = (i + 1) % self.world_size
        self.assigned_params_per_rank = assigned_params_per_rank
        self.params = assigned_params_per_rank[self.rank]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none)


def run_OSS(rank: int, world_size: int, optimizer_cls: Optimizer, use_OSS: bool = True) -> None:
    # d_model = 1600
    # d_ff = 6400
    # num_layers = 48
    # num_heads = 25
    # context_length = 256
    # vocab_size = 10000
    # batch_size = 16
    d_model = 16
    d_ff = 64
    num_layers = 4
    num_heads = 2
    context_length = 256
    vocab_size = 100
    batch_size = 16
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

    model = TransformerLanguageModel(
                d_model,
                num_heads,
                d_ff,
                10000.0,
                vocab_size,
                context_length,
                num_layers,
                device,
            )

    if 'cuda' in device:
        print(f'max allocated(after model init): {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB')
        torch.cuda.reset_peak_memory_stats()

    if use_OSS:
        optimizer = ShardedOptimizer(model.parameters(), optimizer_cls)
    else:
        optimizer = optimizer_cls(model.parameters())
        
    if 'cuda' in device:
        print(f'max allocated(after optimizer init): {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB')
        torch.cuda.reset_peak_memory_stats()

    input_ids = torch.randint(0, vocab_size, (batch_size, context_length))
    label_ids = torch.randint(0, vocab_size, (batch_size, context_length))

    steps = 50
    torch.cuda.synchronize()
    total_time_start = timeit.default_timer()
    step_range = range(steps)
    if rank == 0: 
        step_range = tqdm(step_range)
    for _ in step_range:
        optimizer.zero_grad()
        lm_head_output = model(input_ids)
        loss = cross_entropy(lm_head_output, label_ids)
        loss.backward()

    if 'cuda' in device:
        print(f'max allocated(before optimizer.step): {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB')
        torch.cuda.reset_peak_memory_stats()

        optimizer.step()

    if 'cuda' in device:
        print(f'max allocated(after optimizer.step): {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB')
        torch.cuda.reset_peak_memory_stats()

        permutation = torch.randperm(batch_size)
        input_ids = input_ids[permutation]
        label_ids = label_ids[permutation]
    torch.cuda.synchronize()
    total_time_end = timeit.default_timer()
    total_time_per_step = (total_time_end - total_time_start) / steps
    
    if rank == 0:
        print(f'total time per step: {total_time_per_step:6f}')

    dist.destroy_process_group()


def benchmark_OSS(
    optimizer_cls: Optimizer = AdamW,
    world_size: int = 2,
    use_OSS: bool = True
) -> None:
    mp.spawn(run_OSS, args=(world_size, optimizer_cls, use_OSS), nprocs=world_size, join=True)


if __name__ == '__main__':
    benchmark_OSS(AdamW, 2, True)
    benchmark_OSS(AdamW, 2, False)
    pass