from __future__ import annotations

import math
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError
