import math
import numpy as np
import numpy.typing as npt
import os
import torch
import typing
from collections.abc import Callable
from dataclasses import dataclass, fields
from jaxtyping import Float, Int
from torch import Tensor
from typing import Optional, Iterable


'''
    Given a tensor of inputs and targets, compute the average cross-entropy loss across examples.

    Args:
        inputs (Float[Tensor, 'batch_size vocab_size']): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, 'batch_size']): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, '']: The average cross-entropy loss across examples.
'''
def cross_entropy(inputs: Float[Tensor, ' batch_size vocab_size'], targets: Int[Tensor, ' batch_size']) -> Float[Tensor, '']:
    if inputs.ndim == 3 and targets.ndim == 2:
        inputs = inputs.reshape(-1, inputs.shape[-1])
        targets = targets.reshape(-1)
    stable_inputs = inputs - inputs.max(dim=-1, keepdim=True).values # subtract the largest element for numerical stability
    negative_log_likelihood = stable_inputs.exp().sum(dim=-1).log() - stable_inputs[torch.arange(inputs.shape[0]), targets]
    return negative_log_likelihood.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params, 
        lr: float = 1e-3, 
        weight_decay: float = 0.01, 
        betas: tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8, 
    ):
        if lr < 0:
            raise ValueError(f'Invalid learning rate: {lr}')

        defaults = {
            'lr': lr,
            'weight_decay': weight_decay,
            'beta_1': betas[0],
            'beta_2': betas[1],
            'eps': eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get('t', 1) # Get iteration number from the state, or initial value.
                m = state.get('m', torch.zeros_like(p))
                v = state.get('v', torch.zeros_like(p))
                g = p.grad.data # Get the gradient of loss with respect to p.
                
                m = beta_1 * m + (1 - beta_1) * g # update the first moment estimate
                v = beta_2 * v + (1 - beta_2) * g ** 2 # update the second moment estimate
                lr *= (1 - beta_2 ** t) ** 0.5 / (1 - beta_1 ** t) # compute adjusted lr for iteration t

                p.data -= lr * m / (v ** 0.5 + eps) # update the parameters
                p.data *= 1 - group['lr'] * weight_decay # apply weight decay

                state['t'] = t + 1 # Increment iteration number.
                state['m'] = m
                state['v'] = v

        return loss


'''
    The cosine annealing learning rate schedule takes 
        (i) the current iteration t, 
        (ii) the maximum learning rate alpha_max, 
        (iii) the minimum (final) learning rate alpha_min, 
        (iv) the number of warm-up iterations T_w, and 
        (v) the number of cosine annealing iterations T_c.
'''
'''
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
'''
def cosine_lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        return (
            min_learning_rate 
            + 0.5 
            * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) 
            * (max_learning_rate - min_learning_rate)
        )
    elif it > cosine_cycle_iters:
        return min_learning_rate


def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += (parameter.grad.data ** 2).sum()
    l2_norm = total ** 0.5
    if l2_norm > max_l2_norm:
        epsilon = 1e-6
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.data *= max_l2_norm / (l2_norm + epsilon)


'''
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
'''
def get_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = dataset.shape[0]
    input_start_indices = torch.randint(0, n - context_length, (batch_size,))
    input_end_indices = input_start_indices + context_length

    input_start_indices = input_start_indices.tolist()
    input_end_indices = input_end_indices.tolist()

    inputs = [ dataset[start:end] for start, end in zip(input_start_indices, input_end_indices) ]
    labels = [ dataset[start+1:end+1] for start, end in zip(input_start_indices, input_end_indices) ]

    inputs = np.array(inputs) # convert to numpy arrays first to avoid warnings when creating torch tensors
    labels = np.array(labels) # convert to numpy arrays first to avoid warnings when creating torch tensors

    inputs = torch.tensor(inputs, device=device, dtype=torch.int)
    labels = torch.tensor(labels, device=device, dtype=torch.int)

    return inputs, labels


'''
from assignment_1 p.41:
A common first step when developing any neural net architecture is to overfit to a single minibatch. If
your implementation is correct, you should be able to quickly drive the training loss to near-zero.
'''
def get_single_batch(
    dataset: npt.NDArray, 
    batch_size: int, # this parameter is (intentionally) not used
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    input_start_indices = [0] # fix index to 0
    input_end_indices = [0 + context_length]

    inputs = [ dataset[start:end] for start, end in zip(input_start_indices, input_end_indices) ]
    labels = [ dataset[start+1:end+1] for start, end in zip(input_start_indices, input_end_indices) ]

    inputs = np.array(inputs) # convert to numpy arrays first to avoid warnings when creating torch tensors
    labels = np.array(labels) # convert to numpy arrays first to avoid warnings when creating torch tensors

    inputs = torch.tensor(inputs, device=device, dtype=torch.int)
    labels = torch.tensor(labels, device=device, dtype=torch.int)

    return inputs, labels


def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    entire_state = {
        'model_state': model_state,
        'optimizer_state': optimizer_state,
        'iteration': iteration
    }
    torch.save(entire_state, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer | None = None
) -> int:
    entire_state = torch.load(src)
    model.load_state_dict(entire_state['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(entire_state['optimizer_state'])
    iteration = entire_state['iteration']

    return iteration


'''
    Problem (training_together): Put it together (4 points)
    Deliverable:
    Write a script that runs a training loop to train your model on user-provided input.
    In particular, we recommend that your training script allow for (at least) the following:
    • Ability to configure and control the various model and optimizer hyperparameters.
    • Memory-efficient loading of training and validation large datasets with np.memmap.
    • Serializing checkpoints to a user-provided path.
    • Periodically logging training and validation performance (e.g., to console and/or an external
    service like Weights and Biases).
'''
def train_and_save_tokenizer(
    bpe_train_corpus_path: str,
    vocab_size: int,
    special_tokens: list[str],
    tokenizer_save_path: str
) -> int:
    vocab, merges = train_bpe(bpe_train_corpus_path, vocab_size - 1, special_tokens)
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    torch.save(tokenizer, tokenizer_save_path)

    return len(tokenizer.vocab)


def multiprocess_tokenize_and_save_corpus_ids(
    tokenizer_path: str,
    corpus_path: str,
    corpus_ids_save_path: str,
    num_processes: int = 1,
    max_cache_len: int = 0
) -> None:
    import multiprocessing
    from cs336_basics.pretokenization_example import find_chunk_boundaries

    tokenizer = torch.load(tokenizer_path, weights_only=False)

    with open(corpus_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, '<|endoftext|>'.encode('utf-8'))

        def tokenize(chunk, output_queue: multiprocessing.Queue, use_tqdm: bool = False) -> None:
            chunk_ids = []
            np_chunk_ids = np.array([], dtype=np.uint16)

            for id in tokenizer.encode_iterable(chunk, use_tqdm, max_cache_len):
                chunk_ids.append(id)
                if len(chunk_ids) == 1024 * 1024:
                    np_chunk_ids = np.concatenate((np_chunk_ids, np.array(chunk_ids, dtype=np.uint16)))
                    chunk_ids = []

            if len(chunk_ids) != 0:
                np_chunk_ids = np.concatenate((np_chunk_ids, np.array(chunk_ids, dtype=np.uint16)))
            
            output_queue.put(np_chunk_ids)

        output_queue = multiprocessing.Queue()
        processes = []
        use_tqdm = True # only the first process uses tqdm to display progress
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode('utf-8', errors='ignore')
            process = multiprocessing.Process(target=tokenize, args=(chunk, output_queue, use_tqdm))
            process.start()
            processes.append(process)
            use_tqdm = False # only the first process uses tqdm to display progress

        corpus_ids = np.array([], dtype=np.uint16)
        for _ in processes:
            chunk_ids = output_queue.get()
            corpus_ids = np.concatenate((corpus_ids, chunk_ids))

        for process in processes:
            process.join()

        corpus_ids = np.array(corpus_ids, dtype=np.uint16)
        np.save(corpus_ids_save_path, corpus_ids)


def inspect_sample():
    context_length = 256
    tokenizer = torch.load(config.tokenizer_path, weights_only=False)

    corpus_ids = np.load(config.corpus_ids_path + '.npy', mmap_mode='r')
    input_ids, label_ids = get_batch(corpus_ids, 1, context_length, 'cpu')

    print(tokenizer.decode(input_ids.tolist()[0]))
    print()
    print(tokenizer.decode(label_ids.tolist()[0]))


def load_and_generate(
    context_length: int,
    checkpoint_file_name: str, # file name must include the file extension if it exists
    end_token: str,
    prompt: str,
    completion_only: bool = False,
    max_generation_tokens: int | None = None,
    temparature: float = 1.0,
    top_p: float = 1.0
) -> None:
    model = TransformerLanguageModel(*TransformerLanguageModelConfig(context_length=context_length).get_config())
    load_checkpoint(config.checkpoint_dir + checkpoint_file_name, model)
    tokenizer = torch.load(config.tokenizer_path, weights_only=False)
    end_token_id = tokenizer.encode(end_token)[0]
    input_ids = tokenizer.encode(prompt)
    output = model.generate(
        input_ids, 
        end_token_id, 
        completion_only=completion_only,
        max_generation_tokens=max_generation_tokens,
        temparature=temparature,
        top_p=top_p
    )
    print(tokenizer.decode(output))


@dataclass
class Config:
    def get_config(self):
        return (getattr(self, field.name) for field in fields(self))


@dataclass
class TransformerLanguageModelConfig(Config):
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    theta: float = 10000.0
    vocab_size: int = 10000
    context_length: int = 256
    num_layers: int = 4
    device: str = 'cuda'
    dtype: torch.dtype = torch.float32


@dataclass
class CosineLRScheduleConfig(Config):
    it: int = 0
    max_learning_rate: float = 1e-3
    min_learning_rate: float = 0.0
    warmup_iters: int | None = None
    cosine_cycle_iters: int = 1000
    def __post_init__(self):
        if self.warmup_iters is None:
            self.warmup_iters = int(0.1 * self.cosine_cycle_iters)


@dataclass
class GradientClippingConfig(Config):
    max_l2_norm: float = 1.0
    l2_norm_logging_steps: int | None = None


@dataclass
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    context_length: int
    lr: float = 1e-3
    batch_size: int = 64
    validation_batch_size: int = 64
    steps: int = 10
    validation_steps: int | None = None
    lr_schedule_config: CosineLRScheduleConfig | None = None
    gradient_clipping_config: GradientClippingConfig | None = None
    checkpoint_load_path: str | None = None
    save_steps: int | None = None
    use_wandb: bool = True

    def train(self):
        assert self.save_steps != 0
        assert self.validation_steps != 0
        assert self.gradient_clipping_config is None or self.gradient_clipping_config.l2_norm_logging_steps != 0

        if self.checkpoint_load_path is None:
            iteration = 1
        if self.checkpoint_load_path is not None:
            iteration = load_checkpoint(self.checkpoint_load_path, self.model, self.optimizer)

        corpus_ids = np.load(config.corpus_ids_path + '.npy', mmap_mode='r')
        validation_corpus_ids = np.load(config.validation_corpus_ids_path + '.npy', mmap_mode='r')

        if self.use_wandb:
            wandb.login()
            run = wandb.init(
                project='cs336-assignment1',
                name=f'lr{self.lr:.0e}_ablation_{self.model.ablation_mode}',
                config={
                    'learning rate': self.lr,
                    'steps': self.steps,
                },
            )

        while iteration <= self.steps:
            if self.save_steps is not None and iteration % self.save_steps == 0:
                checkpoint_save_path = config.checkpoint_dir + f'iteration-{iteration}'
                save_checkpoint(self.model, self.optimizer, iteration, checkpoint_save_path)

            if self.lr_schedule_config is not None:
                self.lr_schedule_config.it = iteration
                self.optimizer.param_groups[0]['lr'] = cosine_lr_schedule(*self.lr_schedule_config.get_config())
            self.optimizer.zero_grad()

            input_ids, label_ids = get_batch(corpus_ids, self.batch_size, self.context_length, device)
            lm_head_output = self.model.forward(input_ids)
            loss = cross_entropy(lm_head_output, label_ids)
            print(' '*50, end='\r')
            print(f'step {iteration} / {self.steps}, loss = {loss.cpu().item():.4f}', end='\r')

            loss.backward()

            if (self.gradient_clipping_config is not None 
                and self.gradient_clipping_config.l2_norm_logging_steps is not None
                and iteration % self.gradient_clipping_config.l2_norm_logging_steps == 0):
                total = 0.0
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        total += (parameter.grad.data ** 2).sum()
                l2_norm = total ** 0.5
                if self.use_wandb:
                    wandb.log(
                        {
                            'gradient l2 norm': l2_norm
                        }, 
                        step=iteration
                    )

            if self.gradient_clipping_config is not None and self.gradient_clipping_config.max_l2_norm > 0:
                clip_gradient(self.model.parameters(), self.gradient_clipping_config.max_l2_norm)
            self.optimizer.step()

            if self.use_wandb:
                wandb.log(
                    {
                        'train loss': loss,
                        'learning rate': self.optimizer.param_groups[0]['lr']
                    }, 
                    step=iteration
                )

            if self.validation_steps is not None and iteration % self.validation_steps == 0:
                self.model.eval()
                validation_input_ids, validation_label_ids = get_batch(validation_corpus_ids, self.validation_batch_size, self.context_length, device) 
                # todo: modify to perform validation on the entire validation corpus
                with torch.no_grad():
                    lm_head_output = self.model.forward(validation_input_ids)
                    loss = cross_entropy(lm_head_output, validation_label_ids)
                    print(' '*50, end='\r')
                    print(f'validation loss = {loss.cpu().item():.4f}', end='\r')
                self.model.train()

                if self.use_wandb:
                    wandb.log(
                        {
                            'validation loss': loss
                        }, 
                        step=iteration
                    )

            iteration += 1
        
        wandb.finish()

        iteration -= 1
        checkpoint_save_path = config.checkpoint_dir + f'iteration-{iteration}'
        save_checkpoint(self.model, self.optimizer, iteration, checkpoint_save_path)


def ensure_checkpoint_dir_exists():
    checkpoint_dir = 'cs336_basics/checkpoint'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    

if __name__ == '__main__':
    import os
    import config
    import wandb
    from cs336_basics.bpe import Tokenizer, train_bpe
    from cs336_basics.transformer_language_model import TransformerLanguageModel

    # todo: modify to accept hyperparameters via argparse

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')

    ensure_checkpoint_dir_exists()

    vocab_size = 10240
    special_tokens = []

    # train_and_save_tokenizer(config.bpe_train_corpus_path, vocab_size, special_tokens, config.tokenizer_path)

    # multiprocess_tokenize_and_save_corpus_ids(config.tokenizer_path, config.validation_corpus_path, config.validation_corpus_ids_path, num_processes=12, max_cache_len=1024)
    # multiprocess_tokenize_and_save_corpus_ids(config.tokenizer_path, config.corpus_path, config.corpus_ids_path, num_processes=12, max_cache_len=1024)

    lr = 1e-3
    context_length = 256
    batch_size = 128
    validation_batch_size = 128
    total_tokens_processed = 327680000
    steps = int(total_tokens_processed / batch_size / context_length) // 2
    validation_steps = int(0.02 * steps)
    print(f'steps: {steps}')
    print(f'validation steps: {validation_steps}')

    for ablation_mode in [None, 'without_rmsnorm', 'postnorm', 'without_rope', 'silu']:
        if ablation_mode == 'silu':
            model = TransformerLanguageModel(
                *TransformerLanguageModelConfig(
                    context_length=context_length,
                    d_ff = 4 * 512
                ).get_config(), 
                ablation_mode=ablation_mode
            )
        else:
            model = TransformerLanguageModel(*TransformerLanguageModelConfig(context_length=context_length).get_config(), ablation_mode=ablation_mode)
        model = torch.compile(model)

        optimizer = AdamW(model.parameters(), lr)

        cosine_lr_schedule_config = CosineLRScheduleConfig(
            max_learning_rate = lr,
            cosine_cycle_iters = steps
        )
        gradient_clipping_config = GradientClippingConfig(
            max_l2_norm = 10.0,
            l2_norm_logging_steps = int(0.02 * steps)
        )
        trainer = Trainer(
            model, # pyright: ignore
            optimizer,
            context_length,
            lr,
            batch_size,
            validation_batch_size,
            steps,
            validation_steps,
            cosine_lr_schedule_config,
            gradient_clipping_config,
            use_wandb = True
        )
        trainer.train()