import math
import torch
from torch import Tensor


class FlashAttentionForwardPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            Q: Tensor,  # (B, H, N, D)
            K: Tensor,  # (B, H, N, D)
            V: Tensor,  # (B, H, N, D)
            is_causal: bool = False,
    ) -> Tensor:  # (B, H, N, D)
        raise NotImplementedError

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError