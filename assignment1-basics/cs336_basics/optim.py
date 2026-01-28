from __future__ import annotations
from collections.abc import Callable
from typing import Optional
import math
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,                  # α
        betas: tuple[float, float] = (0.9, 0.999),  # (β1, β2)
        eps: float = 1e-8,                 # ϵ
        weight_decay: float = 0.0,         # λ
    ):
        super().__init__()

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        raise NotImplementedError
