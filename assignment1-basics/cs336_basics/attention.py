from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum

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

    raise NotImplementedError

