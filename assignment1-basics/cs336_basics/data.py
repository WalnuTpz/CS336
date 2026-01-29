from __future__ import annotations
import numpy as np
import torch


def get_batch(    # 取一批数据
    x: np.ndarray,  # (N,)
    batch_size: int,  # B
    context_length: int,  # T
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:  # ((B, T), (B, T))
    N = len(x)
    starts = torch.randint(0, N - context_length, (batch_size, ), device="cpu")    # 起点
    offset = torch.arange(context_length + 1)    # 相对偏移量

    idx = starts[:, None] + offset[None, :]  # (B, 1) + (1, T + 1) -> (B, T + 1)
    x_t = torch.as_tensor(x, dtype=torch.long)  # (N,)
    seq = x_t[idx]    # 在 x 中用 idx 这个索引矩阵取出对应的值

    inputs = seq[:,0 : context_length]    # 输入序列
    targets = seq[:, 1 : context_length + 1]    # 目标序列
    inputs = inputs.to(device=device)
    targets = targets.to(device=device)

    return (inputs, targets)

