from __future__ import annotations
import torch
from typing import BinaryIO, IO, Union
import os


def save_checkpoint(    # 数据保存
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    model_state = model.state_dict()
    opt_state = optimizer.state_dict()
    checkpoint = {"model": model_state, "optimizer": opt_state, "iteration": iteration}    # 模型状态，优化器状态，迭代步数
    torch.save(checkpoint, out)    # 将 obj 保存到 out 中


def load_checkpoint(    # 数据加载
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")    # 从 src 中读取 checkpoint
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]
