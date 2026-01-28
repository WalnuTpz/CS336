from __future__ import annotations

from collections.abc import Callable
from typing import Optional
import math
import torch
from torch import Tensor
from collections.abc import Iterable


class AdamW(torch.optim.Optimizer):    # AdamW 优化器
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

def lr_cosine_schedule(    # 带 warmup 的 cosine 学习率调度
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    if t < T_w:
        alpha_t = t * 1.0 / T_w * alpha_max
    elif t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        cos = math.cos(progress * math.pi)
        alpha_t = alpha_min + 1.0 / 2 * (1 + cos) * (alpha_max - alpha_min)
    else:
        alpha_t = alpha_min

    return  alpha_t

def gradient_clipping(    # 梯度裁剪
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    params:list[torch.nn.Parameter] = list(parameters)
    total_sq: float = 0    # 梯度总和

    for p in params:
        if p.grad is not None:    # 只考虑梯度不为零的参数
            g = p.grad.detach().float()
            total_sq += g.pow(2).sum()
    total_norm = math.sqrt(total_sq)    # 梯度的 L2 范数

    scale = max_l2_norm / (total_norm + eps)
    scale = min(scale, 1.0)

    for p in params:    # 将所有梯度进行裁剪
        if p.grad is not None:
            p.grad.mul_(scale)

    return