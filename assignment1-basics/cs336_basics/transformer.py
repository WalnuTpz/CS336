from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from cs336_basics.layers import RMSNorm, SwiGLUFFN
from cs336_basics.attention import MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block (RMSNorm -> sublayer -> residual), with:
      1) causal multi-head self-attention
      2) SwiGLU FFN
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: nn.Module | None = None,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        super().__init__()

    def forward(
        self,
        x: Tensor,  # (batch, seq, d_model)
        token_positions: Tensor | None = None,  # (batch, seq)
    ) -> Tensor:

        raise NotImplementedError