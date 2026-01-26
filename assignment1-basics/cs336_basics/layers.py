from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor, einsum

class Linear(nn.Module):
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
        y = einsum("... d_in, d_out d_in -> ... d_out", x, self.weight)     # 线性层操作 y = weight * x
        return y

class Embedding(nn.Module):
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
        return self.weight[token_ids]      # 返回词表中对应的行向量

class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps

        w = torch.ones((dim, ), device=device, dtype=dtype)
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> Tensor:
        x_float = x.float()     # 先转换为浮点数，防止爆精度
        rms = torch.sqrt(torch.mean(x_float.pow(2), dim=-1, keepdim=True) + self.eps)     # rms 函数的分母
        x_norm = x / rms.to(x.dtype)
        x_rmsnorm = einsum("... d, d -> ... d", x_norm, self.weight)     # rmsnorm(x) = x_norm · weight
        return x_rmsnorm

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
