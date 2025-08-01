import cs336_basics
import torch
import torch.cuda.nvtx as nvtx
import timeit
from contextlib import nullcontext
from cs336_basics.transformer_language_model import TransformerLanguageModel, softmax
from cs336_basics.trainer import AdamW, TransformerLanguageModelConfig, cross_entropy
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
    '''
        benchmark results on A100:
            forward_pass_only=False
            do_compile=False
                seq_len     256       1024      4096      8192      16384
                d_model                                                  
                16       0.003059  0.002463  0.022000  0.081238  0.325173
                32       0.002087  0.002555  0.022442  0.083447  0.333559
                64       0.002075  0.002441  0.023391  0.087596  0.349094
                128      0.002068  0.002496  0.025365  0.095230  0.378991

            forward_pass_only=True
            do_compile=False
                seq_len     256       1024      4096      8192      16384
                d_model                                                  
                16       0.000943  0.000567  0.006108  0.022698  0.090291
                32       0.000240  0.000532  0.006332  0.023609  0.093935
                64       0.000242  0.000562  0.006788  0.025441  0.101489
                128      0.000241  0.000622  0.007653  0.029206  0.116211

            forward_pass_only=True
            do_compile=True
                seq_len     256       1024      4096      8192      16384
                d_model                                                  
                16       0.000238  0.000481  0.003631  0.013679  0.052226
                32       0.000335  0.000448  0.004353  0.015626  0.056281
                64       0.000306  0.000673  0.004820  0.016214  0.063881
                128      0.000303  0.000733  0.005683  0.020036  0.078846
    '''

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')

    run_benchmark_attention(
        forward_pass_only=True,
        do_compile=True
    )


# uv run nsys profile -o result --force-overwrite true python cs336_systems/benchmark.py
# \\wsl$\Ubuntu-22.04