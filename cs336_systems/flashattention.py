import torch
import triton
import triton.language as tl
from contextlib import nullcontext
from cs336_basics.transformer_language_model import softmax
from torch import inf


def flash_fwd_get_configs(pre_hook=None):
    return [
        triton.Config({'Q_TILE_SIZE': TILE_SIZE, 'K_TILE_SIZE': TILE_SIZE, 'is_causal': True}) # changing the value of 'is_causal' allows all tests to pass
        for TILE_SIZE in [16, 32, 64, 128]
    ]


@triton.autotune(
    configs=flash_fwd_get_configs(),
    key=['N_QUERIES', 'D'],
)
# uv run pytest -k test_flash_forward_pass_triton
# after adding autotune, only one of the two tests passes because 'is_causal' is fixed.
# changing the value of 'is_causal' allows all tests to pass
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale, # 1 / d ** 0.5
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, # B_q
    K_TILE_SIZE: tl.constexpr, # B_k
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    T_k = (N_KEYS + K_TILE_SIZE - 1) // K_TILE_SIZE # cdiv

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
    O_i_j = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i_j = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i_j = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    m_i_j_pre = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

    for j in range(1, T_k + 1):
        # K_j = K[:, (j - 1) * B_k : j * B_k, :] # (batch_size, B_k, d)
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')

        # V_j = V[:, (j - 1) * B_k : j * B_k, :] # (batch_size, B_k, d)
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

        # S_i_j = Q_i @ K_j.transpose(-2, -1) / d ** 0.5 # (batch_size, B_q, B_k)
        S_i_j = tl.dot(Q_i, K_j.trans(1, 0)) * scale

        if is_causal:
            Q_indices = tl.arange(0, Q_TILE_SIZE)[:, None].broadcast_to(Q_TILE_SIZE, K_TILE_SIZE)
            Q_indices += query_tile_index * Q_TILE_SIZE
            K_indices = tl.arange(0, K_TILE_SIZE)[None, :].broadcast_to(Q_TILE_SIZE, K_TILE_SIZE)
            K_indices += (j - 1) * K_TILE_SIZE
            causal_mask = tl.where(Q_indices >= K_indices, 0, 1)
            causal_mask *= -1e6
            S_i_j += causal_mask

        # m_i_j_pre = m_i_j.detach().clone() # m_i_j_pre is m_i_(j-1)
        m_i_j_pre = m_i_j

        # m_i_j = torch.max(m_i_j, torch.max(S_i_j, dim=-1).values) # (batch_size, B_q)
        m_i_j = tl.maximum(m_i_j, tl.max(S_i_j, axis=-1))

        # m_i_j.unsqueeze_(-1) # (batch_size, B_q, 1)
        # P_hat_i_j = torch.exp(S_i_j - m_i_j) # (batch_size, B_q, B_k)
        # m_i_j.squeeze_(-1) # (batch_size, B_q)
        P_hat_i_j = tl.exp(S_i_j - m_i_j[:, None])

        # l_i_j = torch.exp(m_i_j_pre - m_i_j) * l_i_j + torch.sum(P_hat_i_j, dim=-1) # (batch_size, B_q)
        l_i_j = tl.exp(m_i_j_pre - m_i_j) * l_i_j + tl.sum(P_hat_i_j, axis=-1)

        # O_i_j = torch.diag_embed(tl.exp(m_i_j_pre - m_i_j)) @ O_i_j + P_hat_i_j @ V_j # (batch_size, B_q, d)
        O_i_j = tl.exp(m_i_j_pre - m_i_j)[:, None] * O_i_j + tl.dot(P_hat_i_j.cast(V_j.dtype), V_j)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    L_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    # O_i = torch.inverse(torch.diag_embed(l_i_j)) @ O_i_j # (batch_size, B_q, d)
    inverse = tl.div_rn(tl.full((Q_TILE_SIZE,), 1.0, dtype=tl.float32), l_i_j)[:, None].broadcast_to(Q_TILE_SIZE, D)
    O_i = inverse * O_i_j
    
    # L_i = m_i_j + torch.log(l_i_j) # (batch_size, B_q)
    L_i = m_i_j + tl.log(l_i_j)

    # O[:, (i - 1) * B_q : i * B_q, :] = O_i
    # L[:, (i - 1) * B_q : i * B_q] = L_i
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        '''
        Q: (batch_size, N_q, d)
        K: (batch_size, N_k, d)
        V: (batch_size, N_k, d)
        '''
        assert len(Q.shape) == 3 and len(K.shape) == 3 and len(V.shape) == 3
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V)

        ctx.batch_size = Q.shape[0]
        ctx.N_q = Q.shape[-2]
        ctx.N_k = K.shape[-2]
        ctx.d = Q.shape[-1]
        # ctx.B_q = 16
        # ctx.B_k = 16
        # ctx.T_q = (ctx.N_q + ctx.B_q - 1) // ctx.B_q # cdiv
        # ctx.T_k = (ctx.N_k + ctx.B_k - 1) // ctx.B_k # cdiv

        O = torch.empty(ctx.batch_size, ctx.N_q, ctx.d, device=Q.device)
        L = torch.empty(ctx.batch_size, ctx.N_q, device=Q.device)

        grid = lambda META: (triton.cdiv(ctx.N_q, META['Q_TILE_SIZE']), ctx.batch_size)

        # flash_fwd_kernel[(ctx.T_q, ctx.batch_size)](
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2), 
            K.stride(0), K.stride(1), K.stride(2), 
            V.stride(0), V.stride(1), V.stride(2), 
            O.stride(0), O.stride(1), O.stride(2), 
            L.stride(0), L.stride(1), 
            ctx.N_q, ctx.N_k,
            1 / ctx.d ** 0.5,
            ctx.d,
            # ctx.B_q,
            # ctx.B_k,
            # is_causal
        )

        O = O.to(Q.dtype)
        L = L.to(Q.dtype)

        ctx.save_for_backward(Q, K, V, O, L)

        return O

    @staticmethod
    def backward(ctx, dO):
        '''
        Q: (batch_size, N_q, d)
        K: (batch_size, N_k, d)
        V: (batch_size, N_k, d)
        '''
        Q, K, V, O, L = ctx.saved_tensors
        N_q = ctx.N_q
        N_k = ctx.N_k
        # B_q = ctx.B_q
        # B_k = ctx.B_k
        # T_q = ctx.T_q
        # T_k = ctx.T_k

        device = torch.device('cuda')

        batch_size = Q.shape[0]
        d = Q.shape[-1]

        S = Q @ K.transpose(-2, -1) / d ** 0.5

        # P = torch.empty(batch_size, N_q, N_k, device=device)
        # for i in range(1, T_q + 1):
        #     for j in range(1, T_k + 1):
        #         S_i_j = S[:, (i - 1) * B_q : i * B_q, (j - 1) * B_k : j * B_k]
        #         L_i = L[:, (i - 1) * B_q : i * B_q]
        #         P[:, (i - 1) * B_q : i * B_q, (j - 1) * B_k : j * B_k] = torch.exp(S_i_j - L_i.unsqueeze(-1))
        P = torch.exp(S - L.unsqueeze(-1))

        dV = P.transpose(-2, -1) @ dO

        dP = dO @ V.transpose(-2, -1)

        D = torch.sum(O * dO, dim=-1)

        # dS = torch.empty(batch_size, N_q, N_k, device=device)
        # for i in range(1, T_q + 1):
        #     for j in range(1, T_k + 1):
        #         P_ij = P[:, (i - 1) * B_q : i * B_q, (j - 1) * B_k : j * B_k]
        #         dP_ij = dP[:, (i - 1) * B_q : i * B_q, (j - 1) * B_k : j * B_k]
        #         D_i = D[:, (i - 1) * B_q : i * B_q]
        #         dS[:, (i - 1) * B_q : i * B_q, (j - 1) * B_k : j * B_k] = P_ij * (dP_ij - D_i.unsqueeze(-1))
        dS = P * (dP - D.unsqueeze(-1))
        
        dQ = dS @ K / d ** 0.5

        dK = dS.transpose(-2, -1) @ Q / d ** 0.5

        return dQ, dK, dV, None


'''
Problem (flash_forward): 15 points
    (a) Write a pure PyTorch (no Triton) autograd.Function that implements the FlashAttention-2
        forward pass. This will be a lot slower than the regular PyTorch implementation, but will help
        you debug your Triton kernel.
        Your implementation should take input Q, K, and V as well as a flag is_causal and produce
        the output O and the logsumexp value L. You can ignore the is_causal flag for this task. The
        autograd.Function forward should then use save L, Q, K, V, O for the backward pass and
        return O. Remember that the implementation of the forward method of autograd.Function
        always takes the context as its first parameter. Any autograd.Function class needs to
        implement a backward method, but for now you can make it just raise NotImplementedError.
        If you need something to compare against, you can implement Equation 4 to 6 and 12 in
        PyTorch and compare your outputs.
        The interface is then def forward(ctx, Q, K, V, is_causal=False). Determine your own
        tile sizes, but make sure they are at least of size 16 x 16. We will always test your code with
        dimensions that are clean powers of 2 and at least 16, so you don't need to worry about
        out-of-bounds accesses.
'''
class FlashAttention2_PyTorch(torch.autograd.Function):
    # uv run pytest -k test_flash_forward_pass_pytorch
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        '''
        Q: (batch_size, N_q, d)
        K: (batch_size, N_k, d)
        V: (batch_size, N_k, d)
        '''
        assert len(Q.shape) == 3
        assert len(K.shape) == 3
        assert len(V.shape) == 3

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        Q = Q.to(device)
        K = K.to(device)
        V = V.to(device)

        batch_size = Q.shape[0]
        d = Q.shape[-1]
        N_q = Q.shape[-2]
        N_k = K.shape[-2]
        B_q = 16
        B_k = 16
        # B_q = 2
        # B_k = 2
        T_q = (N_q + B_q - 1) // B_q # cdiv
        T_k = (N_k + B_k - 1) // B_k # cdiv
        

        # naive attention for comparing O values
        naive_S = Q @ K.transpose(-2, -1) / d ** 0.5 # (batch_size, N_q, N_k)
        if is_causal:
            mask = torch.tril(torch.ones(N_q, N_k)).bool().cuda()
            naive_S.masked_fill_(~mask, -inf) # (batch_size, N_q, N_k)
        naive_P = softmax(naive_S, dim=-1) # (batch_size, N_q, N_k)
        naive_O = naive_P @ V # (batch_size, N_q, d)


        # pytorch version FA2 (not triton)
        O = torch.empty(batch_size, N_q, d, device=device)
        L = torch.empty(batch_size, N_q, device=device)

        for i in range(1, T_q + 1):
            Q_i = Q[:, (i - 1) * B_q : i * B_q, :] # (batch_size, B_q, d)
            O_i_j = torch.zeros(batch_size, B_q, d, device=device)
            l_i_j = torch.zeros(batch_size, B_q, device=device)
            m_i_j = torch.full((batch_size, B_q), -inf, device=device)
            for j in range(1, T_k + 1):
                K_j = K[:, (j - 1) * B_k : j * B_k, :] # (batch_size, B_k, d)
                V_j = V[:, (j - 1) * B_k : j * B_k, :] # (batch_size, B_k, d)
                S_i_j = Q_i @ K_j.transpose(-2, -1) / d ** 0.5 # (batch_size, B_q, B_k)
                m_i_j_pre = m_i_j.detach().clone() # m_i_j_pre is m_i_(j-1)
                m_i_j = torch.max(m_i_j, torch.max(S_i_j, dim=-1).values) # (batch_size, B_q)
                m_i_j.unsqueeze_(-1) # (batch_size, B_q, 1)
                P_hat_i_j = torch.exp(S_i_j - m_i_j) # (batch_size, B_q, B_k)
                m_i_j.squeeze_(-1) # (batch_size, B_q)
                l_i_j = torch.exp(m_i_j_pre - m_i_j) * l_i_j + torch.sum(P_hat_i_j, dim=-1) # (batch_size, B_q)
                O_i_j = torch.diag_embed(torch.exp(m_i_j_pre - m_i_j)) @ O_i_j + P_hat_i_j @ V_j # (batch_size, B_q, d)

            O_i = torch.inverse(torch.diag_embed(l_i_j)) @ O_i_j # (batch_size, B_q, d)
            L_i = m_i_j + torch.log(l_i_j) # (batch_size, B_q)

            O[:, (i - 1) * B_q : i * B_q, :] = O_i
            L[:, (i - 1) * B_q : i * B_q] = L_i

        # assert torch.allclose(O, naive_O)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.N_q = N_q
        ctx.N_k = N_k
        ctx.B_q = B_q
        ctx.B_k = B_k
        ctx.T_q = T_q
        ctx.T_k = T_k

        return O

    # uv run pytest -k test_flash_backward
    @staticmethod
    def backward(ctx, dO):
        '''
        Q: (batch_size, N_q, d)
        K: (batch_size, N_k, d)
        V: (batch_size, N_k, d)
        '''
        Q, K, V, O, L = ctx.saved_tensors
        N_q = ctx.N_q
        N_k = ctx.N_k
        B_q = ctx.B_q
        B_k = ctx.B_k
        T_q = ctx.T_q
        T_k = ctx.T_k

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')

        batch_size = Q.shape[0]
        d = Q.shape[-1]

        S = Q @ K.transpose(-2, -1) / d ** 0.5

        # P = torch.empty(batch_size, N_q, N_k, device=device)
        # for i in range(1, T_q + 1):
        #     for j in range(1, T_k + 1):
        #         S_i_j = S[:, (i - 1) * B_q : i * B_q, (j - 1) * B_k : j * B_k]
        #         L_i = L[:, (i - 1) * B_q : i * B_q]
        #         P[:, (i - 1) * B_q : i * B_q, (j - 1) * B_k : j * B_k] = torch.exp(S_i_j - L_i.unsqueeze(-1))
        P = torch.exp(S - L.unsqueeze(-1))

        dV = P.transpose(-2, -1) @ dO

        dP = dO @ V.transpose(-2, -1)

        D = torch.sum(O * dO, dim=-1)

        # dS = torch.empty(batch_size, N_q, N_k, device=device)
        # for i in range(1, T_q + 1):
        #     for j in range(1, T_k + 1):
        #         P_ij = P[:, (i - 1) * B_q : i * B_q, (j - 1) * B_k : j * B_k]
        #         dP_ij = dP[:, (i - 1) * B_q : i * B_q, (j - 1) * B_k : j * B_k]
        #         D_i = D[:, (i - 1) * B_q : i * B_q]
        #         dS[:, (i - 1) * B_q : i * B_q, (j - 1) * B_k : j * B_k] = P_ij * (dP_ij - D_i.unsqueeze(-1))
        dS = P * (dP - D.unsqueeze(-1))
        
        dQ = dS @ K / d ** 0.5

        dK = dS.transpose(-2, -1) @ Q / d ** 0.5

        return dQ, dK, dV, None


if __name__ == '__main__':
    batch_size = 16
    N_q = 32
    N_k = 32
    d = 64
    Q = torch.arange(batch_size * N_q * d, dtype=torch.float).reshape(batch_size, N_q, d).cuda()
    K = torch.arange(batch_size * N_q * d, batch_size * N_q * d + batch_size * N_k * d, dtype=torch.float).reshape(batch_size, N_k, d).cuda()
    V = torch.arange(batch_size * N_q * d + batch_size * N_k * d, batch_size * N_q * d + 2 * batch_size * N_k * d, dtype=torch.float).reshape(batch_size, N_k, d).cuda()

    torch_O = (FlashAttention2_PyTorch.apply)(Q, K, V)
    triton_O = (FlashAttention2.apply)(Q, K, V, True)