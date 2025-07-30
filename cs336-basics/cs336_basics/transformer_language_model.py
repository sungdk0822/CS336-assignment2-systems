import torch
from torch import inf, nn
from torch.nn.init import trunc_normal_
from typing import Literal
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None): 
        '''
        Construct a linear transformation module. 
        This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        sigma = (2 / (in_features + out_features)) ** 0.5
        self.W = nn.Parameter(
            trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype, requires_grad=True), 
                mean=0,
                std=sigma,
                a=-3 * sigma,
                b=3 * sigma
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Apply the linear transformation to the input.'''
        return einsum(x, self.W, '... d_in, d_out d_in -> ... d_out')


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        '''
        Construct an embedding module. This function should accept the following parameters:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.embedding = nn.Parameter(
            trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype, requires_grad=True), 
                mean=0,
                std=1,
                a=-3,
                b=3
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''Lookup the embedding vectors for the given token IDs.'''
        return self.embedding[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        Construct the RMSNorm module. This function should accept the following parameters:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype, requires_grad=True))
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.'''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = ( (x ** 2).sum(dim=-1) / self.d_model + self.eps ) ** 0.5
        result = x * self.g.reshape(1, 1, self.d_model) / rms.unsqueeze_(dim=-1)

        return result.to(in_dtype)


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.W_1 = Linear(d_model, d_ff, device, dtype)
        self.W_2 = Linear(d_ff, d_model, device, dtype)
        self.W_3 = Linear(d_model, d_ff, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_2( SiLU(self.W_1(x)) * (self.W_3(x)) )


'''
    from assignment_1 p.44:
    Ablation 3: SwiGLU vs. SiLU
    Next, we will follow Shazeer [2020] and test the importance of gating
    in the feed-forward network, by comparing the performance of SwiGLU feed-forward networks versus feed-
    forward networks using SiLU activations but no gated linear unit (GLU):
'''
class FFN_SiLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.W_1 = Linear(d_model, d_ff, device, dtype)
        self.W_2 = Linear(d_ff, d_model, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_2( SiLU( self.W_1(x) ) )


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        Construct the RoPE module and create buffers if needed.
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        '''
        super().__init__()

        i_indices = torch.arange(max_seq_len)
        k_indices = torch.arange(0, d_k // 2)
        i_grid, k_grid = torch.meshgrid(i_indices, k_indices, indexing='ij')

        theta_values = i_grid / theta ** (2 * k_grid / d_k) # (max_seq_len, d_k // 2)
        cos_values = torch.cos(theta_values) # (max_seq_len, d_k // 2)
        sin_values = torch.sin(theta_values) # (max_seq_len, d_k // 2)

        self.register_buffer(name='cos', tensor=cos_values, persistent=False)
        self.register_buffer(name='sin', tensor=sin_values, persistent=False)

        self.cos = self.cos.to(device=device)
        self.sin = self.sin.to(device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions.
        You should assume that the token positions are a tensor of shape (..., seq_len) 
        specifying the token positions of x along the sequence dimension.
        '''
        cos_values = self.cos[token_positions] # self.cos: (max_seq_len, d_k // 2) -> cos_values: (seq_len, d_k // 2) # pyright: ignore
        sin_values = self.sin[token_positions] # self.sin: (max_seq_len, d_k // 2) -> sin_values: (seq_len, d_k // 2) # pyright: ignore
        
        x_even = x[..., ::2] # (batch_size, seq_len, d_k // 2)
        x_odd = x[..., 1::2] # (batch_size, seq_len, d_k // 2)

        rotated_x_even = x_even * cos_values - x_odd * sin_values # (batch_size, seq_len, d_k // 2)
        rotated_x_odd = x_even * sin_values + x_odd * cos_values # (batch_size, seq_len, d_k // 2)

        rotated_x_even.unsqueeze_(dim=-2) # (batch_size, seq_len, 1, d_k // 2)
        rotated_x_odd.unsqueeze_(dim=-2) # (batch_size, seq_len, 1, d_k // 2)

        rotated_x = torch.cat([rotated_x_even, rotated_x_odd], dim=-2) # (batch_size, seq_len, 2, d_k // 2)
        rotated_x = rotated_x.transpose(-2, -1) # (batch_size, seq_len, d_k // 2, 2)
        rotated_x = rotated_x.reshape(*x.shape) # (batch_size, seq_len, d_k)

        return rotated_x


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert -x.ndim <= dim < x.ndim
    if dim < 0:
        dim += x.ndim
    x -= torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x)

    return exp_x / exp_x.sum(dim=dim, keepdim=True)


'''
Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points)

Deliverable: Implement the scaled dot-product attention function. Your implementation should
handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
(batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
dimensions (if provided). The implementation should return an output with the shape (batch_size,
..., d_v). See section 3.3 for a discussion on batch-like dimensions.
Your implementation should also support an optional user-provided boolean mask of shape (seq_len,
seq_len). The attention probabilities of positions with a mask value of True should collectively sum
to 1, and the attention probabilities of positions with a mask value of False should be zero.
'''
def scaled_dot_product_attention(
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
    pre_softmax = queries @ keys.transpose(-2, -1) / d_k ** 0.5 # (batch_size, ..., n, m)
    if mask is not None:
        pre_softmax.masked_fill_(~mask, -inf) # (batch_size, ..., n, m)
    post_softmax = softmax(pre_softmax, dim=-1) # (batch_size, ..., n, m)

    return post_softmax @ values # (batch_size, ..., n, d_v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        if rope is not None:
            self.rope = rope
        self.d_k = self.d_v = d_model // num_heads
        self.W_Q = Linear(d_model, d_model, device, dtype)
        self.W_K = Linear(d_model, d_model, device, dtype)
        self.W_V = Linear(d_model, d_model, device, dtype)
        self.W_O = Linear(d_model, d_model, device, dtype)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        Q = self.W_Q.forward(x) # (batch_size, seq_len, d_model)
        K = self.W_K.forward(x) # (batch_size, seq_len, d_model)
        V = self.W_V.forward(x) # (batch_size, seq_len, d_model)
        Q = Q.reshape(*Q.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2) # (batch_size, num_heads, seq_len, d_k)
        K = K.reshape(*K.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2) # (batch_size, num_heads, seq_len, d_k)
        V = V.reshape(*V.shape[:-1], self.num_heads, self.d_v).transpose(-3, -2) # (batch_size, num_heads, seq_len, d_v)

        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(device=self.device)
        pre_concat = scaled_dot_product_attention(Q, K, V, causal_mask) # (batch_size, num_heads, seq_len, d_v)
        post_concat = pre_concat.transpose(-3, -2).reshape_as(x) # (batch_size, seq_len, d_model)

        return self.W_O.forward(post_concat) # (batch_size, seq_len, d_model)

    def forward_with_rope(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        assert self.rope is not None
        Q = self.W_Q.forward(x) # (batch_size, seq_len, d_model)
        K = self.W_K.forward(x) # (batch_size, seq_len, d_model)
        V = self.W_V.forward(x) # (batch_size, seq_len, d_model)
        Q = Q.reshape(*Q.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2) # (batch_size, num_heads, seq_len, d_k)
        K = K.reshape(*K.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2) # (batch_size, num_heads, seq_len, d_k)
        V = V.reshape(*V.shape[:-1], self.num_heads, self.d_v).transpose(-3, -2) # (batch_size, num_heads, seq_len, d_v)

        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len)
        # apply RoPE
        Q = self.rope.forward(Q, token_positions)
        K = self.rope.forward(K, token_positions)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(device=self.device)
        pre_concat = scaled_dot_product_attention(Q, K, V, causal_mask) # (batch_size, num_heads, seq_len, d_v)
        post_concat = pre_concat.transpose(-3, -2).reshape_as(x) # (batch_size, seq_len, d_model)

        return self.W_O.forward(post_concat) # (batch_size, seq_len, d_model)


class PrenormTransformer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        max_seq_len: int, 
        theta: float, 
        device=None, 
        dtype=None,
        ablation_mode : Literal['without_rmsnorm', 'postnorm', 'without_rope', 'silu'] | None = None
    ):
        super().__init__()
        rope = RoPE(theta, d_model // num_heads, max_seq_len, device)
        self.MHA = MultiHeadSelfAttention(d_model, num_heads, rope, device, dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device, dtype)
        self.rmsnorm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rmsnorm_2 = RMSNorm(d_model, device=device, dtype=dtype)

        if ablation_mode == 'without_rmsnorm':
            self.forward = self.forward_without_rmsnorm
        elif ablation_mode == 'postnorm':
            self.forward = self.forward_postnorm
        elif ablation_mode == 'without_rope':
            self.forward = self.forward_without_rope
        elif ablation_mode == 'silu':
            self.ffn_silu = FFN_SiLU(d_model, d_ff, device, dtype)
            self.forward = self.forward_silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.MHA.forward_with_rope(self.rmsnorm_1.forward(x))
        x = x + self.swiglu.forward(self.rmsnorm_2.forward(x))

        return x

    # forward function without RMSNorm for ablation on training stability
    def forward_without_rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.MHA.forward_with_rope(x)
        x = x + self.swiglu.forward(x)
        
        return x

    # forward function with post-Norm for comparison with pre-Norm
    def forward_postnorm(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rmsnorm_1.forward(x + self.MHA.forward_with_rope(x))
        x = self.rmsnorm_2.forward(x + self.swiglu.forward(x))

        return x

    # forward function without RoPE for ablation on position embeddings
    def forward_without_rope(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.MHA.forward(self.rmsnorm_1.forward(x))
        x = x + self.swiglu.forward(self.rmsnorm_2.forward(x))

        return x

    # forward function with FFN_SiLU for comparison with SwiGLU
    def forward_silu(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.MHA.forward_with_rope(self.rmsnorm_1.forward(x))
        x = x + self.ffn_silu.forward(self.rmsnorm_2.forward(x))

        return x


class TransformerLanguageModel(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        theta: float, 
        vocab_size: int, 
        context_length: int, 
        num_layers: int, 
        device = None, 
        dtype = None,
        ablation_mode : Literal['', 'without_rmsnorm', 'postnorm', 'without_rope', 'silu'] | None = None
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(PrenormTransformer(d_model, num_heads, d_ff, context_length, theta, device, dtype, ablation_mode))
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
        self.context_length = context_length
        self.device = device
        self.dtype = dtype
        if ablation_mode is None:
            ablation_mode = ''
        self.ablation_mode = ablation_mode
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings.forward(x)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)
        
        return x

    def generate(
        self, 
        input_ids: list[int],
        end_token_id: int,
        completion_only: bool = False,
        max_generation_tokens: int | None = None,
        temparature: float = 1.0,
        top_p : float = 1.0
    ):
        initial_prompt_len = len(input_ids)
        assert initial_prompt_len < self.context_length, f'len(input_ids) must be smaller than self.context_length = {self.context_length}'
        assert temparature >= 0.0, 'temperature must be greater than or equal to 0'
        assert 0.0 <= top_p <= 1.0 
        if temparature == 0.0:
            temparature = 1e-6

        if max_generation_tokens is None:
            max_generation_tokens = self.context_length - initial_prompt_len
        if initial_prompt_len + max_generation_tokens > self.context_length:
            max_generation_tokens = self.context_length - initial_prompt_len

        while max_generation_tokens is None or len(input_ids) < initial_prompt_len + max_generation_tokens:
            logits = self.forward(torch.tensor(input_ids, dtype=torch.int, device=self.device))
            logits /= temparature
            probabilities = softmax(logits, dim=-1) # (1, seq_len, vocab_size)
            probabilities = probabilities[0, -1, :] # (vocab_size, )
            if top_p != 1.0:
                sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
                cumsum = torch.cumsum(sorted_probabilities, dim = -1) # (vocab_size, )
                p_index = torch.searchsorted(cumsum, top_p)
                sorted_probabilities = sorted_probabilities[:p_index + 1]
                sorted_probabilities /= cumsum[p_index]
                next_token_index = torch.multinomial(sorted_probabilities, 1)
                next_token_id = sorted_indices[next_token_index].item()
            else:
                next_token_id = torch.multinomial(probabilities, 1).item()
            
            input_ids.append(next_token_id) # pyright: ignore

            if next_token_id == end_token_id or len(input_ids) >= self.context_length:
                break
        
        return input_ids if not completion_only else input_ids[initial_prompt_len:]


# # todo: implement Loss_ExpBal. Device level balancing term -> omitted
# class DeepSeekMoE(nn.Module):
#     def __init__(
#         self,
#         num_shared_experts: int = 2,
#         num_routed_experts: int = 64,
#         num_active_experts: int = 4, # number excluding shared experts

#     ):


if __name__ == '__main__':
    '''
    Problem (transformer_accounting): Transformer LM resource accounting (5 points)
        (a) Consider GPT-2 XL, which has the following configuration:
            vocab_size : 50,257
            context_length : 1,024
            num_layers : 48
            d_model : 1,600
            num_heads : 25
            d_ff : 6,400
        Suppose we constructed our model using this configuration. How many trainable parameters
        would our model have? Assuming each parameter is represented using single-precision floating
        point, how much memory is required to just load this model?
        Deliverable: A one-to-two sentence response.
    '''
    # GPT-2 XL:
    vocab_size = 50257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25
    d_ff = 6400
    trainable_parameters_num = (  vocab_size * d_model # token_embeddings
                                + num_layers * ( 
                                    4 * d_model * d_model # W_QKVO in MHA
                                    + 3 * d_model * d_ff # W_1, W_2, W_3 in swiglu
                                    + d_model # g in rmsnorm_1
                                    + d_model # g in rmsnorm_2
                                    # + context_length * (d_model // num_heads // 2) * 2 # sin, cos cache in rope (nontrainable)
                                ) # layers
                                + d_model # g in ln_final
                                + d_model * vocab_size  ) # W in lm_head

    print(trainable_parameters_num) # 2127057600

    tlm = TransformerLanguageModel(d_model, num_heads, d_ff, 10000.0, vocab_size, context_length, num_layers)
    real_trainable_parameters_num = sum(parameter.numel() for parameter in tlm.parameters() if parameter.requires_grad)
    print(real_trainable_parameters_num) # 2127057600

    '''
        (b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped
        model. How many FLOPs do these matrix multiplies require in total? Assume that our input
        sequence has context_length tokens.
        Deliverable: A list of matrix multiplies (with descriptions), and the total number of FLOPs
        required.
        (c) Based on your analysis above, which parts of the model require the most FLOPs?
        Deliverable: A one-to-two sentence response.
    '''
    # to do ...

    '''
        (d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24
        layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the
        model size increases, which parts of the Transformer LM take up proportionally more or less of
        the total FLOPs?
        Deliverable: For each model, provide a breakdown of model components and its associated
        FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a
        one-to-two sentence description of how varying the model size changes the proportional FLOPs
        of each component.
        (e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one
        forward pass change?
        How do the relative contribution of FLOPs of the model components
        change?
        Deliverable: A one-to-two sentence response.
    '''
    # to do ...