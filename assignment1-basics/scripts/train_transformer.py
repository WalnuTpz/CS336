from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.losses import cross_entropy, perplexity
from cs336_basics.optim import AdamW, lr_cosine_schedule, gradient_clipping
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.transformer import TransformerLM


@dataclass
class TrainConfig:  # 训练配置数据类：集中保存/传递训练所需的所有超参数与路径等配置
    train_data_path: str
    val_data_path: Optional[str]
    dataset_dtype: str

    # 模型超参数
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    eps: float
    dtype: str

    # 训练超参数
    batch_size: int
    max_iters: int
    grad_clip: float

    # AdamW 超参数
    lr_max: float
    lr_min: float
    betas: tuple[float, float]
    adam_eps: float
    weight_decay: float

    # 调度超参数
    warmup_iters: int
    cosine_cycle_iters: int

    # 日志与验证
    log_interval: int
    eval_interval: int
    eval_batches: int

    # checkpoint
    ckpt_dir: Optional[str]
    ckpt_interval: int
    resume_from: Optional[str]

    # 其他
    device: str
    seed: int
    check_vocab_range: bool
    print_model_summary: bool


def parse_args() -> TrainConfig:  # 从命令行解析训练参数并组装成 TrainConfig 供主流程使用
    parser = argparse.ArgumentParser(description="CS336 Assignment1: Train TransformerLM")

    # 数据路径
    parser.add_argument("--train_data", type=str, required=True, help="Path to train tokens (.npy recommended).")
    parser.add_argument("--val_data", type=str, default=None, help="Optional path to val tokens (.npy recommended).")
    parser.add_argument(
        "--dataset_dtype",
        type=str,
        default="int32",
        choices=["uint16", "int32", "int64"],
        help="dtype used to store token IDs on disk (only used for raw memmap; npy keeps dtype).",
    )

    # 模型超参数
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # AdamW 超参数
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    # 学习率调度
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--cosine_cycle_iters", type=int, default=20000)

    # 日志与验证
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=10)

    # checkpoint
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory to save checkpoints. If None, disable.")
    parser.add_argument("--ckpt_interval", type=int, default=500)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume from.")

    # 其他
    parser.add_argument("--device", type=str, default=None, help="e.g. cpu, cuda, cuda:0. Default auto-detect.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check_vocab_range", action="store_true", help="Check dataset token IDs < vocab_size.")
    parser.add_argument("--print_model_summary", action="store_true", help="Print parameter count and exit early? No, just print once.")

    args = parser.parse_args()

    # 自动选择 device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    cfg = TrainConfig(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        dataset_dtype=args.dataset_dtype,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        eps=args.eps,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        grad_clip=args.grad_clip,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        betas=(args.beta1, args.beta2),
        adam_eps=args.adam_eps,
        weight_decay=args.weight_decay,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        ckpt_dir=args.ckpt_dir,
        ckpt_interval=args.ckpt_interval,
        resume_from=args.resume_from,
        device=device,
        seed=args.seed,
        check_vocab_range=args.check_vocab_range,
        print_model_summary=args.print_model_summary,
    )
    return cfg


def _dtype_from_str(dtype_str: str) -> torch.dtype:  # 将字符串形式的 dtype（如 "float16"）映射为 torch.dtype
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_str}")


def _np_dtype_from_str(dtype_str: str) -> np.dtype:  # 将字符串形式的数据集 dtype（如 "int32"）映射为 numpy dtype
    if dtype_str == "uint16":
        return np.uint16
    if dtype_str == "int32":
        return np.int32
    if dtype_str == "int64":
        return np.int64
    raise ValueError(f"Unknown dataset dtype: {dtype_str}")


def load_token_dataset(path: str, dtype_str: str) -> np.ndarray:  # 加载 token-id 数据集：优先用 mmap 以减少内存占用、加快启动
    # 优先处理 .npy：np.load 支持 mmap_mode='r'
    if path.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
        return arr

    # 处理 raw binary：需要 dtype，shape 由文件大小决定（按 1D 读）
    np_dtype = _np_dtype_from_str(dtype_str)
    file_size = os.path.getsize(path)
    itemsize = np.dtype(np_dtype).itemsize
    if file_size % itemsize != 0:
        raise ValueError(f"File size {file_size} is not divisible by dtype itemsize {itemsize}.")
    length = file_size // itemsize
    arr = np.memmap(path, mode="r", dtype=np_dtype, shape=(length,))
    return arr


def maybe_check_vocab(arr: np.ndarray, vocab_size: int, name: str) -> None:  # 快速健全性检查：确认数据集 token id 都在 [0, vocab_size) 内
    # 这里只做一次 max 检查，避免扫描太慢（memmap 仍可能触发磁盘访问）
    mx = int(np.max(arr))
    if mx >= vocab_size:
        raise ValueError(f"{name}: found token id {mx} >= vocab_size {vocab_size}.")


def build_model(cfg: TrainConfig, device: torch.device, dtype: torch.dtype) -> TransformerLM:  # 用配置构建 TransformerLM
    if cfg.d_model % cfg.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        eps=cfg.eps,
        device=device,
        dtype=dtype,
    )
    return model


def count_parameters(model: torch.nn.Module) -> int:  # 统计模型中可训练参数的总数量（用于打印模型规模）
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    cfg: TrainConfig,
) -> tuple[float, float]:  # 在验证集上采样若干 batch，计算平均 loss 与 perplexity
    model.eval()
    losses: list[float] = []

    for _ in range(cfg.eval_batches):
        x, y = get_batch(dataset, cfg.batch_size, cfg.context_length, cfg.device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(float(loss.detach().cpu().item()))

    loss_mean = float(np.mean(losses))
    ppl = float(torch.exp(torch.tensor(loss_mean)).item())
    model.train()
    return loss_mean, ppl


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:  # 将学习率写入 optimizer 的所有 param_groups（就地更新）
    for group in optimizer.param_groups:
        group["lr"] = lr


def main() -> None:  # 主入口：串起数据加载、建模、训练循环、日志/评估与断点保存恢复
    cfg = parse_args()

    # 设置随机种子，保证可复现
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    dtype = _dtype_from_str(cfg.dtype)

    # CPU 上 float16 往往不稳定/不支持，这里做个保护
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        # 中文：为了避免 CPU half 的坑，自动回退到 float32
        print(f"[warn] device=cpu does not fully support {cfg.dtype} well; falling back to float32.")
        dtype = torch.float32

    # 加载数据（memmap）
    train_data = load_token_dataset(cfg.train_data_path, cfg.dataset_dtype)
    val_data = load_token_dataset(cfg.val_data_path, cfg.dataset_dtype) if cfg.val_data_path else None

    # 可选：检查 token id 范围
    if cfg.check_vocab_range:
        maybe_check_vocab(train_data, cfg.vocab_size, "train_data")
        if val_data is not None:
            maybe_check_vocab(val_data, cfg.vocab_size, "val_data")

    # 构建模型与优化器
    model = build_model(cfg, device=device, dtype=dtype).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr_max,  # 初始 lr 先设成 max，后续每步会用 schedule 覆盖
        betas=cfg.betas,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    # 打印模型参数量
    if cfg.print_model_summary:
        n_params = count_parameters(model)
        print(f"model params: {n_params:,}")

    # checkpoint 目录准备
    if cfg.ckpt_dir is not None:
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # 如果需要从 checkpoint 恢复
    start_it = 0
    if cfg.resume_from is not None:
        it_saved = load_checkpoint(cfg.resume_from, model, optimizer)
        # 中文：约定 checkpoint 里存的是“已完成的迭代数”，下一步从 it_saved 开始
        start_it = int(it_saved)

    model.train()

    # 训练循环
    t_last = time.perf_counter()
    for it in range(start_it, cfg.max_iters):
        # 计算当前 lr，并写入 optimizer
        lr = lr_cosine_schedule(
            t=it,
            alpha_max=cfg.lr_max,
            alpha_min=cfg.lr_min,
            T_w=cfg.warmup_iters,
            T_c=cfg.cosine_cycle_iters,
        )
        set_optimizer_lr(optimizer, lr)

        # 采样 batch
        x, y = get_batch(train_data, cfg.batch_size, cfg.context_length, cfg.device)

        # 前向 + loss
        logits = model(x)
        loss = cross_entropy(logits, y)

        # 反向
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪（可关闭：grad_clip <= 0）
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            gradient_clipping(model.parameters(), cfg.grad_clip)

        # 参数更新
        optimizer.step()

        # 日志
        if (it % cfg.log_interval) == 0:
            t_now = time.perf_counter()
            dt = t_now - t_last
            t_last = t_now

            # 中文：吞吐统计（tokens/sec）
            tokens = cfg.batch_size * cfg.context_length
            tok_per_s = tokens / max(dt, 1e-9)

            loss_val = float(loss.detach().cpu().item())
            ppl_val = float(perplexity(torch.tensor(loss_val)).item())
            print(
                f"it={it:06d}  lr={lr:.6g}  loss={loss_val:.6f}  ppl={ppl_val:.3f}  tok/s={tok_per_s:.1f}"
            )

        # 验证
        if val_data is not None and cfg.eval_interval > 0 and (it % cfg.eval_interval) == 0 and it != start_it:
            val_loss, val_ppl = evaluate(model, val_data, cfg)
            print(f"[eval] it={it:06d}  val_loss={val_loss:.6f}  val_ppl={val_ppl:.3f}")

        # 保存 checkpoint
        if cfg.ckpt_dir is not None and cfg.ckpt_interval > 0 and (it % cfg.ckpt_interval) == 0 and it != start_it:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"ckpt_it{it:06d}.pt")
            # 中文：把“当前迭代号”存进去；恢复时从该 it 继续
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"[ckpt] saved: {ckpt_path}")

    # 训练结束后保存一次
    if cfg.ckpt_dir is not None:
        ckpt_path = os.path.join(cfg.ckpt_dir, f"ckpt_final_it{cfg.max_iters:06d}.pt")
        save_checkpoint(model, optimizer, cfg.max_iters, ckpt_path)
        print(f"[ckpt] saved: {ckpt_path}")


if __name__ == "__main__":
    main()
