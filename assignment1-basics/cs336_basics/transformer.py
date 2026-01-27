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
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rope=rope, device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,  # (batch, seq, d_model)
        token_positions: Tensor | None = None,  # (batch, seq)
    ) -> Tensor:
        if token_positions is None:     # 如果没有传入位置，就自动生成
            seq = x.shape[-2]
            # arange 产生张量 (seq,)，然后用 expand 广播成张量 (batch, seq)，也就是让每个 batch 都对应 [0..seq-1]
            token_positions = torch.arange(seq, device=x.device).expand(*x.shape[: -1])

        h = x + self.attn(self.ln1(x), token_positions)     # attention 子层
        y = h + self.ffn(self.ln2(h))       # FFN 子层

        return y

class TransformerLM(nn.Module):
    """
    Full Transformer Language Model:
      token_embeddings -> [num_layers * TransformerBlock] -> ln_final -> lm_head (logits)
    Uses RoPE (no learned absolute position embeddings).
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

    def forward(self, idx: Tensor) -> Tensor:
        raise NotImplementedError