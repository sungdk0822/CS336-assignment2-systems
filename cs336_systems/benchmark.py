import torch
import torch.cuda.nvtx as nvtx
import timeit
from cs336_basics.transformer_language_model import TransformerLanguageModel
from cs336_basics.trainer import TransformerLanguageModelConfig, cross_entropy


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

    random_input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    random_label_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    @nvtx.range('warmup')
    def warmup():
        for _ in range(warmup_steps):
            lm_head_output = model.forward(random_input_ids)
            if not forward_pass_only:
                loss = cross_entropy(lm_head_output, random_label_ids)
                loss.backward()

        torch.cuda.synchronize()

    @nvtx.range('measure')
    def measure():
        with nvtx.range('forward pass'):
            lm_head_output = model.forward(random_input_ids)   
        if not forward_pass_only:
            torch.cuda.synchronize()
            with nvtx.range('backward pass'):
                loss = cross_entropy(lm_head_output, random_label_ids)
                loss.backward()
        
        torch.cuda.synchronize()

    result = timeit.timeit(
        stmt=measure,
        setup=warmup,
        number=measurement_steps,
        globals=globals()
    )

    print(f'{result / measurement_steps:.4f}')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')

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

    warmup_steps = 5
    forward_pass_only = False
    do_compile = False

    for key, value in model_sizes.items():
        print(key)
        benchmark_pass(
            **value, 
            warmup_steps=warmup_steps, 
            forward_pass_only=forward_pass_only,
            do_compile=do_compile,
            device=device
        )

# uv run nsys profile -o result --force-overwrite true --cuda-memory-usage=true python cs336_systems/benchmark.py
# \\wsl$\Ubuntu-22.04