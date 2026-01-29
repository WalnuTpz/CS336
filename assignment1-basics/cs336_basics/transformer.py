from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from cs336_basics.layers import RMSNorm, SwiGLUFFN, Embedding, Linear
from cs336_basics.attention import MultiHeadSelfAttention, RotaryPositionalEmbedding


class TransformerBlock(nn.Module):    # Transformer 块
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
        if token_positions is None:    # 如果没有传入位置，就自动生成
            batch, seq = x.shape[0], x.shape[1]
            token_positions = torch.arange(seq, device=x.device).expand(batch, seq)
            # arange 可以得到张量 (seq,)，然后用 expand 广播成张量 (batch, seq)，也就是让每个 batch 都对应 [0..seq-1]

        h = x + self.attn(self.ln1(x), token_positions)    # attention 子层
        y = h + self.ffn(self.ln2(h))    # FFN 子层

        return y


class TransformerLM(nn.Module):    # Transformer 语言模型
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
        self.context_length = context_length
        d_k = d_model // num_heads

        rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=d_k,
            max_seq_len=context_length,
            device=device,
        )

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope=rope,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        idx: Tensor  # (batch, seq) int
    ) -> Tensor:
        B, T = idx.shape
        x = self.token_embeddings(idx)    # 把 idx 查表变成张量 (batch, seq, d_model)
        token_positions = torch.arange(T, device=idx.device).expand(B, T)    # (batch, seq)

        for block in self.layers:    # 依次通过所有 transformer block
            x = block(x, token_positions=token_positions)

        x = self.ln_final(x)    # 最后进行一次 RMSNorm
        logits = self.lm_head(x)    # 投影到词表后得到 logits: (batch, seq, vocab_size)

        return logits