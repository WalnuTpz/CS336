import math
import torch
from torch import Tensor
from einops import einsum


def _pick_flash_block_sizes(
    N: int,
    D: int,
) -> tuple[int, int]:
    # PyTorch 版本的瓶颈主要来自 Python 层的分块循环，所以这里更激进一些
    if D <= 64:
        block = 128 if N >= 4096 else 64
    elif D <= 128:
        block = 64 if N >= 1024 else 32
    else:
        block = 64 if N >= 4096 else 32
    return block, block


def _flash_backward_pytorch_impl(
    Q: Tensor,  # (..., N, D)
    K: Tensor,  # (..., N, D)
    V: Tensor,  # (..., N, D)
    O: Tensor,  # (..., N, D)
    dO: Tensor,  # (..., N, D)
    L: Tensor,  # (..., N)
    is_causal: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    *batch_dims, N, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    BLOCK_M, BLOCK_N = _pick_flash_block_sizes(N, D)

    # 为了数值稳定，内部统一用 float32
    Qf = Q.float()
    Kf = K.float()
    Vf = V.float()
    Of = O.float()
    dOf = dO.float()
    Lf = L.float()

    D_vec = torch.sum(dOf * Of, dim=-1)  # (..., N), D vector: D_i = sum_d dO_i[d] * O_i[d]
    dQf = torch.zeros_like(Qf)
    dKf = torch.zeros_like(Kf)
    dVf = torch.zeros_like(Vf)

    for j in range(0, N, BLOCK_N):
        j_end = min(j + BLOCK_N, N)
        Kj = Kf[..., j:j_end, :]  # (..., Nj, D)
        Vj = Vf[..., j:j_end, :]  # (..., Nj, D)
        dKj = torch.zeros((*batch_dims, j_end - j, D), device=Q.device, dtype=torch.float32)
        dVj = torch.zeros((*batch_dims, j_end - j, D), device=Q.device, dtype=torch.float32)
        if is_causal:
            k_idx = torch.arange(j, j_end, device=Q.device)

        # 固定当前 kv block，遍历所有 q block 来累计 dK_j 和 dV_j，并顺带更新 dQ
        for i in range(0, N, BLOCK_M):
            i_end = min(i + BLOCK_M, N)
            Qi = Qf[..., i:i_end, :]  # (..., Mi, D)
            dOi = dOf[..., i:i_end, :]  # (..., Mi, D)
            Li = Lf[..., i:i_end]  # (..., Mi)
            Di = D_vec[..., i:i_end]  # (..., Mi)
            if is_causal:
                q_idx = torch.arange(i, i_end, device=Q.device)

            # S_i^(j) = Q_i @ K_j^T / sqrt(D)
            Sij = einsum(Qi, Kj, "... q d, ... k d -> ... q k") * scale
            if is_causal:
                causal_mask = q_idx[:, None] >= k_idx[None, :]
                Sij = torch.where(causal_mask, Sij, torch.full_like(Sij, float("-inf")))

            # P_i^(j) = exp(S_i^(j) - L_i)，dP_i^(j) = dO_i @ V_j^T
            Pij = torch.exp(Sij - Li.unsqueeze(-1))
            dPij = torch.matmul(dOi, Vj.transpose(-2, -1))
            # dS_i^(j) = P_i^(j) * (dP_i^(j) - D_i)
            dSij = Pij * (dPij - Di.unsqueeze(-1))

            dVj = dVj + torch.matmul(Pij.transpose(-2, -1), dOi)
            dKj = dKj + scale * torch.matmul(dSij.transpose(-2, -1), Qi)
            dQf[..., i:i_end, :] = dQf[..., i:i_end, :] + scale * torch.matmul(dSij, Kj)

        dKf[..., j:j_end, :] = dKj
        dVf[..., j:j_end, :] = dVj

    dQ = dQf.to(Q.dtype)
    dK = dKf.to(K.dtype)
    dV = dVf.to(V.dtype)

    return dQ, dK, dV


try:
    _flash_backward_pytorch_compiled = torch.compile(_flash_backward_pytorch_impl)
except Exception:
    _flash_backward_pytorch_compiled = _flash_backward_pytorch_impl


class FlashAttentionForwardPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Tensor,  # (..., N, D)
        K: Tensor,  # (..., N, D)
        V: Tensor,  # (..., N, D)
        is_causal: bool = False,
    ) -> Tensor:  # (..., N, D)
        *batch_dims, N, D = Q.shape  # *batch_dims 可以支持一维或二维的前导维度
        scale = 1.0 / math.sqrt(D)
        BLOCK_M, BLOCK_N = _pick_flash_block_sizes(N, D)

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
            if is_causal:
                q_idx = torch.arange(i, i_end, device=Q.device)

            for j in range(0, N, BLOCK_N):
                # Nj = j_end - j
                j_end = min(j + BLOCK_N, N)
                Kj = Kf[..., j : j_end, :]  # (..., Nj, D)
                Vj = Vf[..., j : j_end, :]  # (..., Nj, D)

                Sij = einsum(Qi, Kj, "... q d, ... k d -> ... q k") * scale  # (..., Mi, Nj)
                if is_causal:
                    k_idx = torch.arange(j, j_end, device=Q.device)
                    causal_mask = q_idx[:, None] >= k_idx[None, :]
                    Sij = torch.where(causal_mask, Sij, torch.full_like(Sij, float("-inf")))
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
    def backward(ctx, *grad_outputs):
        (dO,) = grad_outputs    # 实际上本题的 grad_outputs 只用接受 dO 这一个输出对应的梯度
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = _flash_backward_pytorch_compiled(
            Q, K, V, O, dO, L, ctx.is_causal
        )
        return dQ, dK, dV, None
