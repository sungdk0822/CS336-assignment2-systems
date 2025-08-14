import cs336_basics
import timeit
import torch
import torch.cuda.nvtx as nvtx
import triton
from cs336_basics.transformer_language_model import TransformerLanguageModel, softmax
from cs336_basics.trainer import AdamW, TransformerLanguageModelConfig, cross_entropy
from cs336_systems.flashattention import FlashAttention2
from torch import inf
from typing import Callable, Literal


@nvtx.range('scaled dot product attention')
def annotated_scaled_dot_product_attention(
    queries: torch.Tensor, 
    keys: torch.Tensor, 
    values: torch.Tensor, 
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    '''
    queries: (batch_size, ..., n, d_k)
    keys: (batch_size, ..., m, d_k)
    values: (batch_size, ..., m, d_v)
    mask: (n, m)
    '''
    d_k = queries.shape[-1]
    with nvtx.range('computing attention scores'):
        pre_softmax = queries @ keys.transpose(-2, -1) / d_k ** 0.5 # (batch_size, ..., n, m)
        if mask is not None:
            pre_softmax.masked_fill_(~mask, -inf) # (batch_size, ..., n, m)

    with nvtx.range('computing softmax'):
        post_softmax = softmax(pre_softmax, dim=-1) # (batch_size, ..., n, m)

    with nvtx.range('final matmul'):
        result = post_softmax @ values # (batch_size, ..., n, d_v)

    return result


def benchmark(
    measure: Callable,
    warmup_steps: int = 5,
    measurement_steps: int = 10,
    forward_pass_only: bool = False,
    memory_snapshot_filename: str = 'memory_snapshot.pickle'
) -> tuple[float, float]:
    measured_times = []

    for step in range(warmup_steps + measurement_steps):
        if step < warmup_steps:
            measure(forward_pass_only)

        if step == warmup_steps:
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        if step >= warmup_steps:
            torch.cuda.cudart().cudaProfilerStart()

            nvtx.range_push(f'step {step}')
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            measure(forward_pass_only)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            nvtx.range_pop()

            measured_times.append(end_time - start_time)

    '''
        This will output a file memory_snapshot.pickle that you can load into the following online tool:
        https://pytorch.org/memory_viz
    '''
    torch.cuda.memory._dump_snapshot(memory_snapshot_filename)
    torch.cuda.memory._record_memory_history(enabled=None)

    avg_time = sum(measured_times) / len(measured_times)

    squared_times = [t ** 2 for t in measured_times]
    std_time = ( sum(squared_times) / len(squared_times) - avg_time ** 2 ) ** 0.5

    print(f'avg {avg_time:.4f} std {std_time:.4f}')

    return avg_time, std_time


def benchmark_pass(
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    context_length: int = 256,
    warmup_steps: int = 5,
    measurement_steps: int = 10,
    forward_pass_only: bool = False,
    do_compile: bool = False,
    device: str = 'cuda'
) -> None:
    batch_size = 4
    vocab_size = 10000

    with nvtx.range('define model'):
        model = TransformerLanguageModel(
            *TransformerLanguageModelConfig(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                vocab_size=vocab_size,
                context_length=context_length,
                num_layers=num_layers,
                device=device
            ).get_config()
        )
        if do_compile:
            model = torch.compile(model)

    lr = 1e-3
    with nvtx.range('define optimizer'):
        optimizer = AdamW(model.parameters(), lr)

    with nvtx.range('define input'):
        random_input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        random_label_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    def measure(forward_pass_only: bool):
        if not forward_pass_only:
            optimizer.zero_grad()
        if forward_pass_only:
            model.zero_grad()

        with nvtx.range('forward pass'):
            lm_head_output = model.forward(random_input_ids)   

        if not forward_pass_only:
            with nvtx.range('backward pass'):
                loss = cross_entropy(lm_head_output, random_label_ids)
                loss.backward()

            with nvtx.range('optimizer step'):
                optimizer.step()

    benchmark(
        measure,
        warmup_steps,
        measurement_steps,
        forward_pass_only
    )


def run_benchmark_pass(
    model_size: Literal['small', 'medium', 'large', 'xl', '2.7B'],
    context_length: int = 256,
    warmup_steps: int = 5,
    forward_pass_only: bool = False,
    do_compile: bool = False,
    use_autocast: bool = True,
    mixed_precision_dtype: torch.dtype = torch.bfloat16
) -> None:
    # You can swap your original implementation with the annotated version in your benchmarking script via:
    cs336_basics.transformer_language_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    model_sizes = {
        'small': {
            'd_model': 768,
            'd_ff': 3072,
            'num_layers': 12,
            'num_heads': 12
        },
        'medium': {
            'd_model': 1024,
            'd_ff': 4096,
            'num_layers': 24,
            'num_heads': 16
        },
        'large': {
            'd_model': 1280,
            'd_ff': 5120,
            'num_layers': 36,
            'num_heads': 20
        },
        'xl': {
            'd_model': 1600,
            'd_ff': 6400,
            'num_layers': 48,
            'num_heads': 25
        },
        '2.7B': {
            'd_model': 2560,
            'd_ff': 10240,
            'num_layers': 32,
            'num_heads': 32
        },
    }

    context = torch.autocast(device_type=device, dtype=mixed_precision_dtype) if use_autocast else nullcontext()

    print(model_size)
    model_hyperparameters = model_sizes[model_size]

    with context:
        benchmark_pass(
            **model_hyperparameters, 
            context_length=context_length,
            warmup_steps=warmup_steps, 
            forward_pass_only=forward_pass_only,
            do_compile=do_compile,
            device=device
        )


def benchmark_attention(
    d_model: int,
    seq_len: int,
    warmup_steps: int = 5,
    measurement_steps: int = 100,
    forward_pass_only: bool = False,
    do_compile: bool = False,
    device: str = 'cuda'
) -> tuple[float, float]:
    from cs336_basics.transformer_language_model import scaled_dot_product_attention
    batch_size = 8
    criterion = torch.nn.MSELoss()

    if do_compile:
        scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)

    with nvtx.range('define input'):
        Q = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)
        K = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)
        V = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)
        label = torch.randn((batch_size, seq_len, d_model), device=device)

    def measure(forward_pass_only: bool):
        with nvtx.range('forward pass'):
            output = scaled_dot_product_attention(Q, K, V)

        if not forward_pass_only:
            with nvtx.range('backward pass'):
                loss = criterion(output, label)
                loss.backward()

    def ensure_dir_exists(dir):
        import os
        if not os.path.exists(dir):
            os.makedirs(dir)

    ensure_dir_exists('memory_snapshots')
    ensure_dir_exists('memory_snapshots/attn')
    memory_snapshot_filename=f'memory_snapshots/attn/dmodel{d_model}_seqlen{seq_len}'
    if forward_pass_only:
        memory_snapshot_filename += '_forwardpassonly'
    if do_compile:
        memory_snapshot_filename += '_compile'
    memory_snapshot_filename += '.pickle'

    avg_time, std_time = benchmark(
        measure,
        warmup_steps,
        measurement_steps,
        forward_pass_only,
        memory_snapshot_filename
    )

    return avg_time, std_time


def run_benchmark_attention(
    forward_pass_only: bool = False,
    do_compile: bool = False,
) -> None:
    import pandas as pd
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    # d_models = [16, 32]
    # seq_lens = [256, 1024, 4096]

    results = []
    for d_model in d_models:
        for seq_len in seq_lens:
            avg_time, std_time = benchmark_attention(
                d_model,
                seq_len,
                forward_pass_only=forward_pass_only,
                do_compile=do_compile
            )
            results.append({
                'd_model': d_model,
                'seq_len': seq_len,
                'avg_time': avg_time
            })
    
    df = pd.DataFrame(results)
    pivot_df = df.pivot(index='d_model', columns='seq_len', values='avg_time')
    print(pivot_df)


def benchmark_flashattention(
    d_model: int,
    seq_len: int,
    dtype: torch.dtype,
    forward_pass_only: bool
) -> float:
    batch_size = 1
    is_causal = True

    Q = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=dtype, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=dtype, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=dtype, requires_grad=True)

    flash = torch.compile(FlashAttention2.apply)

    def flash_forward_backward():
        o = flash(Q, K, V, is_causal)
        loss = o.sum()
        loss.backward()

    def flash_forward():
        o = flash(Q, K, V, is_causal)

    if forward_pass_only:
        avg_time = triton.testing.do_bench(flash_forward, rep=100, warmup=10)
    else:
        avg_time = triton.testing.do_bench(flash_forward_backward, rep=100, warmup=10)
    
    return avg_time


def run_benchmark_flashattention(
    dtype: torch.dtype,
    forward_pass_only: bool = False
) -> None:
    import pandas as pd
    from itertools import product
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384, 32768, 65536]

    results = []
    for d_model, seq_len in product(d_models, seq_lens):
        # print(f'd_model {d_model} seq_len {seq_len}')
        avg_time = benchmark_flashattention(
            d_model,
            seq_len,
            dtype,
            forward_pass_only
        )
        results.append({
            'd_model': d_model,
            'seq_len': seq_len,
            'avg_time': avg_time
        })
    
    df = pd.DataFrame(results)
    pivot_df = df.pivot(index='d_model', columns='seq_len', values='avg_time')
    pivot_df = pivot_df.round(3)
    print(pivot_df)


def benchmark_naive_attention(
    d_model: int,
    seq_len: int,
    dtype: torch.dtype,
    forward_pass_only: bool
) -> float:
    from cs336_basics.transformer_language_model import scaled_dot_product_attention
    scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)

    batch_size = 1

    Q = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=dtype, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=dtype, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=dtype, requires_grad=True)

    def forward_backward():
        o = scaled_dot_product_attention(Q, K, V)
        loss = o.sum()
        loss.backward()

    def forward():
        o = scaled_dot_product_attention(Q, K, V)

    if forward_pass_only:
        avg_time = triton.testing.do_bench(forward, rep=100, warmup=10)
    else:
        avg_time = triton.testing.do_bench(forward_backward, rep=100, warmup=10)
    
    return avg_time


def run_benchmark_naive_attention(
    dtype: torch.dtype,
    forward_pass_only: bool = False
) -> None:
    import pandas as pd
    from itertools import product
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384, 32768, 65536]

    results = []
    for d_model, seq_len in product(d_models, seq_lens):
        avg_time = benchmark_naive_attention(
            d_model,
            seq_len,
            dtype,
            forward_pass_only
        )
        results.append({
            'd_model': d_model,
            'seq_len': seq_len,
            'avg_time': avg_time
        })
    
    df = pd.DataFrame(results)
    pivot_df = df.pivot(index='d_model', columns='seq_len', values='avg_time')
    pivot_df = pivot_df.round(3)
    print(pivot_df)


if __name__ == '__main__':
    import torch._dynamo as torchdynamo
    torchdynamo.config.cache_size_limit = 32
    torch.set_float32_matmul_precision('high')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')


    run_benchmark_naive_attention(
        torch.float32,
        True
    )
    '''
    results on A100 (unit: ms):
        seq_len  256    1024   4096   8192   16384   32768   65536
        d_model                                                   
        16       0.034  0.062  0.337  1.077  4.365  17.646  73.239
        32       0.092  0.094  0.283  0.939  4.134  18.013  83.033
        64       0.294  0.295  0.305  1.009  4.430  19.053  77.196
        128      0.099  0.101  0.343  1.124  4.797  20.062  84.471
    '''

    run_benchmark_naive_attention(
        torch.bfloat16,
        True
    )
    '''
    results on A100 (unit: ms):
        seq_len  256    1024   4096   8192   16384   32768   65536
        d_model                                                   
        16       0.057  0.129  0.285  0.751  2.668  11.314  47.290
        32       0.217  0.219  0.251  0.708  2.836  11.341  49.277
        64       0.210  0.215  0.249  0.825  2.980  11.709  53.132
        128      0.092  0.089  0.273  0.905  3.284  12.932  58.997
    '''

    run_benchmark_naive_attention(
        torch.float32,
        False
    )
    '''
    results on A100 (unit: ms):
        seq_len  256    1024   4096   8192    16384   32768
        d_model                                            
        16       0.858  1.853  2.257  3.159  12.835  52.198
        32       2.318  2.122  2.112  3.034  12.522  52.708
        64       1.520  2.064  2.107  3.203  13.279  55.941
        128      1.068  1.674  2.329  3.392  14.092  66.127
    '''
    
    run_benchmark_naive_attention(
        torch.bfloat16,
        False
    )
    '''
    results on A100 (unit: ms):
        seq_len  256    1024   4096   8192   16384   32768    65536
        d_model                                                    
        16       1.274  2.363  2.160  2.190  7.475  31.984  129.563
        32       2.882  3.177  3.124  3.143  7.899  31.729  147.503
        64       1.346  1.722  1.963  2.230  8.243  32.920  140.866
        128      1.622  2.065  2.359  2.558  8.962  35.909  154.373
    '''


    run_benchmark_flashattention(
        torch.float32,
        True
    )
    '''
    results on A100 (unit: ms):
        (only the forward pass is implemented in Triton, the backward pass is implemented in PyTorch)
        seq_len  256    1024   4096   8192   16384   32768   65536
        d_model                                                   
        16       0.012  0.029  0.099  0.201  0.552   1.933   7.423
        32       0.015  0.036  0.125  0.272  0.775   2.741  11.165
        64       0.023  0.073  0.291  0.652  2.435   8.630  32.312
        128      0.027  0.086  0.351  1.331  4.100  14.810  58.941
    '''

    run_benchmark_flashattention(
        torch.bfloat16,
        True
    )
    '''
    results on A100 (unit: ms):
        (only the forward pass is implemented in Triton, the backward pass is implemented in PyTorch)
        seq_len  256    1024   4096   8192   16384  32768   65536
        d_model                                                  
        16       0.017  0.032  0.090  0.165  0.474  1.423   8.240
        32       0.211  0.176  0.174  0.361  0.917  2.043  10.514
        64       0.141  0.144  0.153  0.267  0.739  2.599  15.113
        128      0.172  0.177  0.324  0.829  2.311  6.703  19.919
    '''

    run_benchmark_flashattention(
        torch.float32,
        False
    )
    '''
    results on A100 (unit: ms):
        (only the forward pass is implemented in Triton, the backward pass is implemented in PyTorch)
        seq_len  256    1024   4096   8192    16384   32768
        d_model                                            
        16       1.132  1.673  1.997  1.855   7.001  28.611
        32       2.035  2.365  2.347  2.498   8.546  27.660
        64       2.032  2.423  2.431  3.470   9.624  36.845
        128      2.733  2.398  2.263  4.923  16.023  45.797
    '''

    run_benchmark_flashattention(
        torch.bfloat16,
        False
    )
    '''
    results on A100 (unit: ms):
        (only the forward pass is implemented in Triton, the backward pass is implemented in PyTorch)
        seq_len  256    1024   4096   8192   16384   32768   65536
        d_model                                                   
        16       1.200  1.442  1.612  1.702  4.368  16.018  64.638
        32       1.486  1.583  1.922  2.230  4.797  19.954  68.294
        64       1.957  1.132  2.246  2.370  6.417  18.094  74.532
        128      1.397  1.255  1.394  2.461  9.207  24.168  96.307
    '''


# uv run nsys profile -o result --force-overwrite true python cs336_systems/benchmark.py
# \\wsl$\Ubuntu-22.04