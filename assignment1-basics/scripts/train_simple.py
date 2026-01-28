from __future__ import annotations

from collections.abc import Callable
from typing import Optional
import argparse
import math
import time
import torch


DEFAULT_LR = 0.5    # 可修改参数：lr
DEFAULT_BATCH_SIZE = 64    # 可修改参数：batch_size
DEFAULT_STEPS = 10
DEFAULT_DIM = 16
DEFAULT_SEED = 0


def get_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SGD(torch.optim.Optimizer):
    """
    SGD with a simple lr decay: lr / sqrt(t + 1)
    (matches the assignment writeup example)
    """
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data

                p.data -= (lr / math.sqrt(t + 1)) * grad
                state["t"] = t + 1

        return loss


class LinearRegressor(torch.nn.Module):
    """
    y_hat = x @ w
    w: (D,)
    x: (B, D)
    y_hat: (B,)
    """
    def __init__(self, D: int, device: torch.device):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(D, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=None, help="e.g. cpu, cuda, cuda:0")
    args = parser.parse_args()

    device = get_device(args.device)
    torch.manual_seed(args.seed)

    # Ground-truth linear function weights (fixed)
    true_w = torch.arange(args.dim, dtype=torch.float32, device=device)

    def get_batch(B: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(B, args.dim, device=device)
        y = x @ true_w
        return x, y

    model = LinearRegressor(args.dim, device=device)
    opt = SGD(model.parameters(), lr=args.lr)

    print(f"device={device} lr={args.lr} batch_size={args.batch_size} steps={args.steps} dim={args.dim}")

    t0 = time.perf_counter()
    last_loss = None

    for step in range(args.steps):
        x, y = get_batch(args.batch_size)

        opt.zero_grad()
        pred = model(x)
        loss = mse_loss(pred, y)
        loss.backward()
        opt.step()

        last_loss = loss.detach().cpu().item()
        print(f"step {step:02d} loss={last_loss:.6f}")

    t1 = time.perf_counter()
    print(f"final_loss_after_{args.steps}_steps={last_loss:.6f}")
    print(f"wall_time_sec={t1 - t0:.4f}")


if __name__ == "__main__":
    main()
