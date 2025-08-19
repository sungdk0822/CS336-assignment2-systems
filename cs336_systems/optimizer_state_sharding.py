import torch.distributed as dist
from torch.optim import Optimizer, AdamW
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
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
        super().__init__(params, kwargs)
        self.optimizer = optimizer_cls(self.params, **kwargs)
        print(len(self.params))
        print(len(self.optimizer.param_groups[-1]['params']))

    def step(self, closure = None, **kwargs):
        '''
        Calls the wrapped optimizer's step() method with the pro-
        vided closure and keyword arguments. After updating the parameters, synchronize with the other
        ranks.
        '''
        self.optimizer.step(closure, **kwargs)
        print('34')
        for param in self.params:
            dist.broadcast(param, self.rank)
            print('37')
        print('38')

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
        self.params = assigned_params_per_rank[self.rank]