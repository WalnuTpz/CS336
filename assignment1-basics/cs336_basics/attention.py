from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum

class RotaryPositionalEmbedding(nn.Module):
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
        self.register_buffer("cos", cos, persistent=False)     # 将 cos, sin 注册为不可学习参数，并且不写进state_dict()
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = x[..., ::2]     # x 的偶数项
        x_odd = x[..., 1::2]     # x 的奇数项

        y_even = x_even * cos - x_odd * sin     # 进行二位旋转
        y_odd = x_even * sin + x_odd * cos
        y = torch.stack([y_even, y_odd], dim=-1).reshape_as(x)      # 将旋转后的张量交错放到 y 中

        return y