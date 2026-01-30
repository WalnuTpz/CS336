from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch

from cs336_basics.losses import cross_entropy, perplexity
from cs336_basics.optim import AdamW, lr_cosine_schedule, gradient_clipping
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.transformer import TransformerLM


# ----------------------------
# 配置数据类
# ----------------------------
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

    # experiment log
    exp_name: Optional[str]
    exp_dir: str

    # 其他
    device: str
    seed: int
    check_vocab_range: bool
    print_model_summary: bool


# ----------------------------
# 小工具：dtype 映射 / 数据集加载
# ----------------------------
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


# ----------------------------
# 小工具：experiment log（JSON）
# ----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _make_unique_json_path(exp_dir: str, exp_name: str) -> str:
    # 如果同名文件已存在，自动加后缀避免覆盖
    base = os.path.join(exp_dir, f"{exp_name}.json")
    if not os.path.exists(base):
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(exp_dir, f"{exp_name}_{stamp}.json")


# ----------------------------
# 数据 batch：避免每步把整个数据集 cast 成 long 的大坑
# ----------------------------
def get_batch_fast(  # 取一批数据（只对 batch 做 cast，不对全量数据做 cast）
    x_cpu: torch.Tensor,  # (N,) on CPU, dtype=uint16/int32/int64 都可
    batch_size: int,  # B
    context_length: int,  # T
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:  # ((B, T), (B, T))
    N = x_cpu.numel()
    if N <= context_length:
        raise ValueError(f"Dataset too small: N={N} <= context_length={context_length}")

    # 这里 high 是“开区间”，最大 start = N - context_length - 1，保证能取到 T+1 个 token
    starts = torch.randint(0, N - context_length, (batch_size,), device="cpu")
    offset = torch.arange(context_length + 1, device="cpu")

    idx = starts[:, None] + offset[None, :]  # (B, T+1)
    seq = x_cpu[idx]  # (B, T+1) 仍是原 dtype（例如 int32/uint16）

    # 只对 batch cast 成 long
    seq = seq.to(dtype=torch.long)

    inputs = seq[:, 0:context_length].to(device=device)
    targets = seq[:, 1 : context_length + 1].to(device=device)
    return inputs, targets


# ----------------------------
# 模型/评估/辅助
# ----------------------------
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


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:  # 将学习率写入 optimizer 的所有 param_groups（就地更新）
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset_cpu: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[float, float]:  # 在验证集上采样若干 batch，计算平均 loss 与 perplexity
    model.eval()
    losses: list[float] = []

    for _ in range(cfg.eval_batches):
        x, y = get_batch_fast(dataset_cpu, cfg.batch_size, cfg.context_length, cfg.device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(float(loss.detach().cpu().item()))

    loss_mean = float(np.mean(losses))
    ppl = float(torch.exp(torch.tensor(loss_mean)).item())
    model.train()
    return loss_mean, ppl


# ----------------------------
# 参数解析
# ----------------------------
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
    parser.add_argument("--context_length", type=int, default=512)
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

    # experiment log
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (used as experiments/<name>.json).")
    parser.add_argument("--exp_dir", type=str, default="experiments", help="Directory to write experiment JSON logs.")

    # 其他
    parser.add_argument("--device", type=str, default=None, help="e.g. cpu, cuda, cuda:0. Default auto-detect.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check_vocab_range", action="store_true", help="Check dataset token IDs < vocab_size.")
    parser.add_argument("--print_model_summary", action="store_true", help="Print parameter count once.")

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
        exp_name=args.exp_name,
        exp_dir=args.exp_dir,
        device=device,
        seed=args.seed,
        check_vocab_range=args.check_vocab_range,
        print_model_summary=args.print_model_summary,
    )
    return cfg


# ----------------------------
# 主入口
# ----------------------------
def main() -> None:  # 主入口：串起数据加载、建模、训练循环、日志/评估与断点保存恢复
    cfg = parse_args()

    # experiment name：若不提供，则用时间戳自动生成（保证每次运行都有独立 JSON）
    if cfg.exp_name is None or cfg.exp_name.strip() == "":
        cfg.exp_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    exp_json_path = _make_unique_json_path(cfg.exp_dir, cfg.exp_name)

    # 设置随机种子，保证可复现
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    dtype = _dtype_from_str(cfg.dtype)

    # CPU 上 float16/bfloat16 往往不稳定/不支持，这里做个保护
    effective_dtype = dtype
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        print(f"[warn] device=cpu does not fully support {cfg.dtype} well; falling back to float32.")
        effective_dtype = torch.float32

    # 先写一份“running”的实验日志（即使中途崩了也能知道参数）
    exp_log = {
        "status": "running",
        "started_at_utc": _utc_now_iso(),
        "argv": sys.argv,
        "config": asdict(cfg),
        "resolved": {
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "device": str(device),
            "effective_dtype": str(effective_dtype).replace("torch.", ""),
        },
    }
    _atomic_write_json(exp_json_path, exp_log)
    print(f"[exp] wrote: {exp_json_path}")

    # 训练主流程：失败也会把 status 写进 JSON
    try:
        # 加载数据（memmap）
        train_np = load_token_dataset(cfg.train_data_path, cfg.dataset_dtype)
        val_np = load_token_dataset(cfg.val_data_path, cfg.dataset_dtype) if cfg.val_data_path else None

        # 可选：检查 token id 范围
        if cfg.check_vocab_range:
            maybe_check_vocab(train_np, cfg.vocab_size, "train_data")
            if val_np is not None:
                maybe_check_vocab(val_np, cfg.vocab_size, "val_data")

        # 将数据集转成 CPU tensor（共享底层内存；不做全量 long cast）
        train_cpu = torch.from_numpy(np.asarray(train_np))
        val_cpu = torch.from_numpy(np.asarray(val_np)) if val_np is not None else None

        # 构建模型与优化器
        model = build_model(cfg, device=device, dtype=effective_dtype).to(device)
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

        # 如果需要从 checkpoint 恢复：checkpoint 里存的是“下一步要跑的 iteration”
        start_it = 0
        if cfg.resume_from is not None:
            next_it = load_checkpoint(cfg.resume_from, model, optimizer)
            start_it = int(next_it)
            print(f"[ckpt] resumed from {cfg.resume_from}, next_it={start_it}")

        model.train()

        # 训练循环：tokens/sec 统计（按上次打印以来跑了多少步来算）
        t_last = time.perf_counter()
        last_log_next_it = start_it  # 语义：上次打印之后的“下一步 it”
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
            x, y = get_batch_fast(train_cpu, cfg.batch_size, cfg.context_length, cfg.device)

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

            next_it = it + 1  # 语义：完成本次 step 后，下一步要跑的迭代号

            # 日志：默认每 log_interval 步打印一次；另外第一步也打印一次方便确认在跑
            do_log = (cfg.log_interval > 0 and (next_it % cfg.log_interval) == 0) or (it == start_it)
            if do_log:
                t_now = time.perf_counter()
                dt = t_now - t_last
                t_last = t_now

                steps = max(next_it - last_log_next_it, 1)
                last_log_next_it = next_it

                tokens = steps * cfg.batch_size * cfg.context_length
                tok_per_s = tokens / max(dt, 1e-9)

                loss_val = float(loss.detach().cpu().item())
                ppl_val = float(perplexity(torch.tensor(loss_val)).item())
                print(
                    f"it={it:06d}  lr={lr:.6g}  loss={loss_val:.6f}  ppl={ppl_val:.3f}  tok/s={tok_per_s:.1f}"
                )

            # 验证：每 eval_interval 步评估一次
            if val_cpu is not None and cfg.eval_interval > 0 and (next_it % cfg.eval_interval) == 0:
                val_loss, val_ppl = evaluate(model, val_cpu, cfg)
                print(f"[eval] it={it:06d}  val_loss={val_loss:.6f}  val_ppl={val_ppl:.3f}")

            # 保存 checkpoint：每 ckpt_interval 步保存一次；checkpoint 里写 next_it（恢复时不重复跑）
            if cfg.ckpt_dir is not None and cfg.ckpt_interval > 0 and (next_it % cfg.ckpt_interval) == 0:
                ckpt_path = os.path.join(cfg.ckpt_dir, f"ckpt_it{next_it:06d}.pt")
                save_checkpoint(model, optimizer, next_it, ckpt_path)
                print(f"[ckpt] saved: {ckpt_path}")

        # 训练结束后保存一次（final 也存 next_it = max_iters）
        final_ckpt_path = None
        if cfg.ckpt_dir is not None:
            final_ckpt_path = os.path.join(cfg.ckpt_dir, f"ckpt_final_it{cfg.max_iters:06d}.pt")
            save_checkpoint(model, optimizer, cfg.max_iters, final_ckpt_path)
            print(f"[ckpt] saved: {final_ckpt_path}")

        # 更新 experiment log：finished
        exp_log["status"] = "finished"
        exp_log["finished_at_utc"] = _utc_now_iso()
        exp_log["result"] = {
            "final_it": cfg.max_iters,
            "final_ckpt": final_ckpt_path,
        }
        _atomic_write_json(exp_json_path, exp_log)

    except Exception as e:
        exp_log["status"] = "failed"
        exp_log["finished_at_utc"] = _utc_now_iso()
        exp_log["error"] = {
            "type": type(e).__name__,
            "message": str(e),
        }
        _atomic_write_json(exp_json_path, exp_log)
        raise


if __name__ == "__main__":
    main()
