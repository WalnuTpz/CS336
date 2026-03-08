import math
import torch
from torch import Tensor
from einops import einsum


class FlashAttentionForwardPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            Q: Tensor,  # (..., N, D)
            K: Tensor,  # (..., N, D)
            V: Tensor,  # (..., N, D)
            is_causal: bool = False,
    ) -> Tensor:  # (..., N, D)
        *batch_dims, N, D = Q.shape
        scale = 1.0 / math.sqrt(D)
        BLOCK_M = 16
        BLOCK_N = 16

        # 为了数值稳定，转换为 float32 计算
        Qf = Q.float()
        Kf = K.float()
        Vf = V.float()

        O = torch.zeros((*batch_dims, N, D), device=Q.device, dtype=torch.float32)
        m = torch.full((*batch_dims, N), float("-inf"), device=Q.device, dtype=torch.float32)
        l = torch.zeros((*batch_dims, N), device=Q.device, dtype=torch.float32)

        for i in range(0, N, BLOCK_M):
            i_end = min(i + BLOCK_M, N)
            Mi = i_end - i
            Qi = Qf[..., i : i_end, :]  # (..., Mi, D)
            Oi = torch.zeros((*batch_dims, Mi, D), device=Q.device, dtype=torch.float32)
            mi = torch.full((*batch_dims, Mi), float("-inf"), device=Q.device, dtype=torch.float32)
            li = torch.zeros((*batch_dims, Mi), device=Q.device, dtype=torch.float32)

            for j in range(0, N, BLOCK_N):
                # Nj = j_end - j
                j_end = min(j + BLOCK_N, N)
                Kj = Kf[..., j : j_end, :]  # (..., Nj, D)
                Vj = Vf[..., j : j_end, :]  # (..., Nj, D)

                Sij = einsum(Qi, Kj, "... q d, ... k d -> ... q k") * scale  # (..., Mi, Nj)
                mij = torch.max(Sij, dim=-1).values  # (..., Mi)，这部分的最大值
                m_new = torch.maximum(mi, mij)    # 更新该行的最大值
                alpha = torch.exp(mi - m_new)    # 缩放系数
                Pij = torch.exp(Sij - m_new.unsqueeze(-1))  # (..., Mi, Nj)
                l_new = alpha * li + torch.sum(Pij, dim=-1)

                Oi = alpha.unsqueeze(-1) * Oi + torch.matmul(Pij, Vj)
                mi = m_new
                li = l_new

            # 当前 query block 扫完所有 KV block 后，写回全局
            O[..., i:i_end, :] = Oi / li.unsqueeze(-1)
            m[..., i:i_end] = mi
            l[..., i:i_end] = li

        L = m + torch.log(l)
        O_out = O.to(Q.dtype)

        ctx.save_for_backward(L, Q, K, V, O_out)
        ctx.is_causal = is_causal
        return O_out

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError