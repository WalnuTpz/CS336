from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum


class Linear(nn.Module):    # 线性层（无 bias）
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        self.weight = nn.Parameter(w)
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        y = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")    # y = weight * x
        return y


class Embedding(nn.Module):    # 嵌入层
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        self.weight = nn.Parameter(w)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]    # 返回词表中对应的行向量


class RMSNorm(nn.Module):    # RMS 归一化
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps

        w = torch.ones((dim, ), device=device, dtype=dtype)
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> Tensor:
        x_float = x.float()    # 先转换为浮点数，防止爆精度
        rms = torch.sqrt(torch.mean(x_float.pow(2), dim=-1, keepdim=True) + self.eps)    # rms 函数的分母
        x_norm = x / rms.to(x.dtype)
        out = einsum(x_norm, self.weight, "... d, d -> ... d")    # rmsnorm(x) = x_norm · weight

        return out


def SiLU(x: Tensor) -> Tensor:    # SiLU 函数
    return x * torch.sigmoid(x)

class SwiGLUFFN(nn.Module):    # SwiGLU 门控前馈层
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = int(math.ceil((8.0 / 3.0) * d_model))
            d_ff = ((d_ff + 63) // 64) * 64    # d_ff 为最接近 (8 / 3) * d_model 的 64 的倍数（此处向上取整）

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> Tensor:
        # ffn = w2 * (SiLU(w1 * x) · (w3 * x))
        a = self.w1(x)    # a = w1 * x
        b = self.w3(x)    # b = w3 * x
        gated = SiLU(a) * b
        out = self.w2(gated)    # ffn = w2 * gated

        return out
