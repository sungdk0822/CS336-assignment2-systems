import cs336_basics
import torch
import torch.cuda.nvtx as nvtx
import timeit
from cs336_basics.transformer_language_model import TransformerLanguageModel, softmax
from cs336_basics.trainer import AdamW, TransformerLanguageModelConfig, cross_entropy
from torch import inf


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

    def measure():
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

    measured_times = []

    for step in range(warmup_steps + measurement_steps):
        if step < warmup_steps:
            measure()

        if step >= warmup_steps:
            torch.cuda.cudart().cudaProfilerStart()

            nvtx.range_push(f'step {step}')
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            measure()
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            nvtx.range_pop()

            measured_times.append(end_time - start_time)

    avg_time = sum(measured_times) / len(measured_times)

    squared_times = [t ** 2 for t in measured_times]
    std_time = ( sum(squared_times) / len(squared_times) - avg_time ** 2 ) ** 0.5

    print(f'avg {avg_time:.4f} std {std_time:.4f}')


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


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')

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
        # 'xl': {
        #     'd_model': 1600,
        #     'd_ff': 6400,
        #     'num_layers': 48,
        #     'num_heads': 25
        # },
        # '2.7B': {
        #     'd_model': 2560,
        #     'd_ff': 10240,
        #     'num_layers': 32,
        #     'num_heads': 32
        # },
    }

    context_length = 512
    warmup_steps = 5
    forward_pass_only = False
    do_compile = False

    for key, value in model_sizes.items():
        print(key)
        benchmark_pass(
            **value, 
            context_length=context_length,
            warmup_steps=warmup_steps, 
            forward_pass_only=forward_pass_only,
            do_compile=do_compile,
            device=device
        )

# uv run nsys profile -o result --force-overwrite true python cs336_systems/benchmark.py
# \\wsl$\Ubuntu-22.04