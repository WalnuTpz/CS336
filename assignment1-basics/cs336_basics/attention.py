from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum, rearrange
from cs336_basics.layers import Linear

class RotaryPositionalEmbedding(nn.Module):     # 旋转位置编码
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        j = torch.arange(0, d_k // 2, device=device, dtype=torch.float32)
        inv_freq = (theta ** (-2.0 * j / d_k))      # 计算每一对维度对应的的逆频率
        pos = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)

        angles = einsum(pos, inv_freq, "t, j -> t j")      # 位置和逆频率做外积得到角度
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer("cos", cos, persistent=False)     # 将 cos, sin 注册为不可学习参数
        self.register_buffer("sin", sin, persistent=False)
        # persistent=False 表示参数不写进state_dict()，让它可以根据不同的 d_k, max_seq_len 重新计算

    def forward(
        self,
        x: Tensor,  # (..., seq_len, d_k)
        token_positions: Tensor  # (..., seq_len)
        ) -> Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = x[..., ::2]     # x 的偶数项
        x_odd = x[..., 1::2]     # x 的奇数项

        y_even = x_even * cos - x_odd * sin     # 进行二位旋转
        y_odd = x_even * sin + x_odd * cos
        y = torch.stack([y_even, y_odd], dim=-1).reshape_as(x)      # 将旋转后的张量交错放到 y 中

        return y

def softmax(x: Tensor, dim: int) -> Tensor:     # 软最大函数
    """
    Apply softmax over dimension `dim` with numerical stability.
    Output shape == input shape.
    """
    mx = torch.max(x, dim=dim, keepdim=True).values
    # keepdim=True 表示在进行 max/sum/mean 等操作以后保留被约掉的那一维，它的长度变为 1
    x_shift = x - mx
    exp_x = torch.exp(x_shift)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    out = exp_x / sum_exp_x

    return out

def scaled_dot_product_attention(      # 缩放点积注意力
    Q: Tensor,  # (..., queries, d_k)
    K: Tensor,  # (..., keys, d_k)
    V: Tensor,  # (..., keys, d_v)
    mask: Tensor | None = None,  # (..., queries, keys), bool
) -> Tensor:
    d_k = Q.shape[-1]
    # scores = (Q @ K^T) / sqrt(d_k)
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)

    if mask is not None:     # 对 scores 进行掩码
        scores = scores.masked_fill(~mask, -torch.inf)

    # out = softmax(scores) * V
    P = softmax(scores, dim=-1)
    out = einsum(P, V, "... queries keys, ... keys d_v -> ...  queries d_v")

    return out

class MultiHeadSelfAttention(nn.Module):     # 多头自注意力
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: nn.Module | None = None,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope

        self.w_q = Linear(self.d_model, self.h * self.d_k, device=device, dtype=dtype)
        self.w_k = Linear(self.d_model, self.h * self.d_k, device=device, dtype=dtype)
        self.w_v = Linear(self.d_model, self.h * self.d_k, device=device, dtype=dtype)
        self.w_o = Linear(self.h * self.d_k, self.d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,  # (batch, seq, d_model)
        token_positions: Tensor
    ) -> Tensor:
        batch, seq, d_model = x.shape

        # 把原始权重拆分成每个 head 的权重
        Wq = rearrange(self.w_q.weight, "(h d_k) d_model -> h d_k d_model", h=self.h, d_k=self.d_k)
        Wk = rearrange(self.w_k.weight, "(h d_k) d_model -> h d_k d_model", h=self.h, d_k=self.d_k)
        Wv = rearrange(self.w_v.weight, "(h d_k) d_model -> h d_k d_model", h=self.h, d_k=self.d_k)

        # 计算出 x 对应的 Q, K, V
        Q = einsum(x, Wq, "batch seq d_model, h d_k d_model -> batch h seq d_k")
        K = einsum(x, Wk, "batch seq d_model, h d_k d_model -> batch h seq d_k")
        V = einsum(x, Wv, "batch seq d_model, h d_k d_model -> batch h seq d_k")

        # 将 Q, K 进行旋转位置编码
        if self.rope is not None:
            pos = token_positions
            if pos.dim() == 2:     # 将 pos 从二维扩展到三维，方便对齐
                pos = pos[:, None, :]
            Q = self.rope(Q, pos)
            K = self.rope(K, pos)

        # 构造下三角为 True 的 mask，使第 t 个位置只能关注 <= t 的 token
        casual = torch.tril(torch.ones(seq, seq, device=x.device, dtype=torch.bool))
        mask = casual[None, None, :, :]     # 将 casual 从二维扩展到四维，方便对齐

        out = scaled_dot_product_attention(Q, K, V, mask=mask)      # 调用缩放点积注意力

        Wo = rearrange(self.w_o.weight, "d_model (h d_k) -> d_model h d_k", h=self.h, d_k=self.d_k)
        y = einsum(out, Wo, "batch h seq d_k, d_model h d_k -> batch seq d_model")      # 将多头结果合并回 d_model

        return y