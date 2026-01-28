from __future__ import annotations
from collections.abc import Callable
from typing import Optional
import math
import torch
from torch import Tensor

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,                  # α
        betas: tuple[float, float] = (0.9, 0.999),  # (β1, β2)
        eps: float = 1e-8,                 # ϵ
        weight_decay: float = 0.0,         # λ
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()    # 默认不需要梯度跟踪
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():    # 如果有 closure 函数，用它计算初始 loss
                loss = closure()

        for group in self.param_groups:    # 遍历参数组
            # 加载对应的超参数
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:    # 遍历参数组内的每个张量 p
                if p.grad is None:
                    continue

                g = p.grad.data
                state = self.state[p]

                # 如果是初始状态，则初始化 t, m, v
                if len(state) == 0:
                    state["t"]: int = 0
                    state["m"]: Tensor = torch.zeros_like(p.data)
                    state["v"]: Tensor = torch.zeros_like(p.data)

                # 根据公式更新 t, m, v
                t = state["t"] + 1
                m = beta1 * state["m"] + (1 - beta1) * g
                v = beta2 * state["v"] + (1 - beta2) * (g * g)

                # 根据公式计算 alpha_t 并更新 p 的参数
                alpha_t = lr * math.sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t))
                p.data -= alpha_t * m / (v.sqrt() + eps)
                p.data -= lr * wd * p.data

                # 保存更新后的 t, m, v
                state["t"] = t
                state["m"] = m
                state["v"] = v

        return  loss
