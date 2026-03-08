import math
import torch
from torch import Tensor
from einops import einsum


class FlashAttentionForwardPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            Q: Tensor,  # (B, H, N, D)
            K: Tensor,  # (B, H, N, D)
            V: Tensor,  # (B, H, N, D)
            is_causal: bool = False,
    ) -> Tensor:  # (B, H, N, D)
        D = Q.shape[-1]
        scale = 1.0 / math.sqrt(D)

        S = einsum(Q, K, "... q d, ... k d -> ... q k") * scale  # (B, H, N, N)
        L = torch.logsumexp(S, dim=-1)  # (B, H, N)
        P = torch.softmax(S, dim=-1)  # (B, H, N, N)
        O = einsum(P, V, "... q k, ... k d -> ... q d")  # (B, H, N, D)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError