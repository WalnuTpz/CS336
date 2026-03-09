"""
一个 optimizer state sharding benchmark 脚本：
1) memory：比较带/不带 optimizer state sharding 时的显存占用
2) speed：比较带/不带 optimizer state sharding 时的训练步耗时

默认设置：
- 使用 NaiveDDP 做梯度同步
- 比较 baseline AdamW 与 ShardedOptimizer(AdamW)
- 支持 torchrun 多机启动，也支持单机自动 spawn

用法示例：
  # 单机 2 卡：测 XL 的显存
  uv run torchrun --standalone --nproc_per_node=2 scripts/optimizer_state_sharding_benchmark.py --mode memory --size xl

  # 单机 2 卡：测多种模型规模的训练步耗时
  uv run torchrun --standalone --nproc_per_node=2 scripts/optimizer_state_sharding_benchmark.py --mode speed --sizes small medium large xl
"""

from __future__ import annotations

import argparse
import os
import socket
import timeit
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import NaiveDDP
from cs336_systems.sharded_optim import ShardedOptimizer


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


SIZE_PRESETS = {
    "small": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelSpec(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelSpec(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": ModelSpec(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def find_free_port() -> int:    # 找一个本机可用端口，供单机 spawn 初始化进程组使用。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def sync_if_cuda(device: torch.device) -> None:    # 只在 CUDA 上做同步，保证计时和显存采样边界清晰。
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(
    *,
    vocab_size: int,
    context_length: int,
    rope_theta: float,
    spec: ModelSpec,
) -> BasicsTransformerLM:    # 按给定规格构造 Transformer 语言模型。
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=rope_theta,
    )


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:    # 把 [B, T, V] logits 转成一个标量 loss。
    batch_size, context_length, vocab_size = logits.shape
    return F.cross_entropy(
        logits.reshape(batch_size * context_length, vocab_size),
        targets.reshape(batch_size * context_length),
    )


def choose_spec(
    size: str,
) -> ModelSpec:    # 根据 size preset 选择模型规格。
    return SIZE_PRESETS[size]


def launched_with_torchrun() -> bool:    # 判断当前进程是否由 torchrun 启动。
    return all(name in os.environ for name in ["RANK", "WORLD_SIZE", "LOCAL_RANK"])


def init_process_group_from_env() -> tuple[int, int, int, torch.device]:    # 从 torchrun 环境变量初始化进程组。
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=30),
    )
    return rank, world_size, local_rank, device


def build_optimizer(
    impl: str,
    params,
    lr: float,
) -> torch.optim.Optimizer:    # 按实现名选择普通 AdamW 或 sharded optimizer。
    if impl == "baseline":
        return AdamW(params, lr=lr)
    if impl == "sharded":
        return ShardedOptimizer(
            params,
            AdamW,
            lr=lr,
        )
    raise ValueError(f"unknown impl: {impl}")


def reduce_max_scalar(
    value: float,
    device: torch.device,
) -> float:    # 取所有 rank 中的最大标量，更符合一次分布式训练步的真实完成时间/最大显存。
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def collect_memory_stats(
    device: torch.device,
) -> tuple[float, float]:    # 读取当前 allocated 和到目前为止的 peak allocated，单位为 GiB。
    alloc_gib = torch.cuda.memory_allocated(device) / (1024 ** 3)
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return alloc_gib, peak_gib


def run_memory_profile(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    impl: str,
    args: argparse.Namespace,
    spec: ModelSpec,
) -> dict[str, Any]:    # 跑一个训练步，记录初始化后、step 前、step 后的显存占用。
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.cuda.empty_cache()
    sync_if_cuda(device)

    model = build_model(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        spec=spec,
    ).to(device=device, dtype=dtype)
    ddp_model = NaiveDDP(model)
    optimizer = build_optimizer(
        impl,
        ddp_model.parameters(),
        args.lr,
    )
    if isinstance(optimizer, ShardedOptimizer):
        optimizer.enable_timing = True

    local_batch_size = args.local_batch_size
    context_length = args.context_length
    vocab_size = args.vocab_size

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(local_batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    y = torch.roll(x, shifts=-1, dims=1)

    sync_if_cuda(device)
    torch.cuda.reset_peak_memory_stats(device)
    after_init_alloc_gib, after_init_peak_gib = collect_memory_stats(device)

    optimizer.zero_grad(set_to_none=True)
    dist.barrier()
    sync_if_cuda(device)

    logits = ddp_model(x)
    loss = compute_loss(logits.float(), y)
    loss.backward()
    ddp_model.finish_gradient_synchronization()

    sync_if_cuda(device)
    before_step_alloc_gib, before_step_peak_gib = collect_memory_stats(device)

    optimizer.step()
    sync_if_cuda(device)
    after_step_alloc_gib, after_step_peak_gib = collect_memory_stats(device)

    row = {
        "impl": impl,
        "world_size": world_size,
        "size": args.size,
        "d_model": spec.d_model,
        "after_init_alloc_gib": reduce_max_scalar(after_init_alloc_gib, device),
        "after_init_peak_gib": reduce_max_scalar(after_init_peak_gib, device),
        "before_step_alloc_gib": reduce_max_scalar(before_step_alloc_gib, device),
        "before_step_peak_gib": reduce_max_scalar(before_step_peak_gib, device),
        "after_step_alloc_gib": reduce_max_scalar(after_step_alloc_gib, device),
        "after_step_peak_gib": reduce_max_scalar(after_step_peak_gib, device),
    }

    del loss, logits, x, y, optimizer, ddp_model, model
    torch.cuda.empty_cache()
    sync_if_cuda(device)
    dist.barrier()
    return row


def run_speed_benchmark(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    impl: str,
    args: argparse.Namespace,
    spec: ModelSpec,
    size_name: str,
) -> dict[str, Any]:    # 计时完整训练步，并单独统计参数更新通信时间。
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.cuda.empty_cache()
    sync_if_cuda(device)

    model = build_model(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        spec=spec,
    ).to(device=device, dtype=dtype)
    ddp_model = NaiveDDP(model)
    optimizer = build_optimizer(
        impl,
        ddp_model.parameters(),
        args.lr,
    )

    local_batch_size = args.local_batch_size
    context_length = args.context_length
    vocab_size = args.vocab_size

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(local_batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    y = torch.roll(x, shifts=-1, dims=1)

    def run_one_step() -> tuple[float, float]:
        optimizer.zero_grad(set_to_none=True)
        dist.barrier()
        sync_if_cuda(device)

        timer = timeit.default_timer
        t0 = timer()

        logits = ddp_model(x)
        loss = compute_loss(logits.float(), y)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()

        sync_if_cuda(device)
        t1 = timer()

        param_update_comm_ms = 0.0
        if isinstance(optimizer, ShardedOptimizer):
            param_update_comm_ms = optimizer.last_sync_ms
        return (t1 - t0) * 1000.0, param_update_comm_ms

    for _ in range(args.warmup_steps):
        run_one_step()

    step_times_ms: list[float] = []
    param_update_comm_times_ms: list[float] = []

    for _ in range(args.steps):
        local_step_ms, local_param_update_comm_ms = run_one_step()
        step_times_ms.append(reduce_max_scalar(local_step_ms, device))
        param_update_comm_times_ms.append(reduce_max_scalar(local_param_update_comm_ms, device))

    step_times = np.array(step_times_ms, dtype=np.float64)
    param_update_comm_times = np.array(param_update_comm_times_ms, dtype=np.float64)
    mean_step_ms = float(step_times.mean())
    std_step_ms = float(step_times.std(ddof=1)) if len(step_times) > 1 else 0.0
    mean_param_update_comm_ms = float(param_update_comm_times.mean())
    comm_ratio = (mean_param_update_comm_ms / mean_step_ms) if mean_step_ms > 0 else 0.0
    global_tokens_per_step = world_size * local_batch_size * context_length
    throughput = global_tokens_per_step / (mean_step_ms / 1000.0)

    row = {
        "impl": impl,
        "world_size": world_size,
        "size": size_name,
        "d_model": spec.d_model,
        "d_ff": spec.d_ff,
        "num_layers": spec.num_layers,
        "num_heads": spec.num_heads,
        "avg_step_ms": mean_step_ms,
        "std_step_ms": std_step_ms,
        "param_update_comm_ms": mean_param_update_comm_ms,
        "param_update_comm_ratio": comm_ratio,
        "throughput_tok_s": throughput,
    }

    del optimizer, ddp_model, model, x, y
    torch.cuda.empty_cache()
    sync_if_cuda(device)
    dist.barrier()
    return row


def print_memory_table(
    rows: list[dict[str, Any]],
) -> None:    # 打印 optimizer state sharding 的显存对比表。
    print("========== Optimizer State Sharding Memory ==========")
    print(
        f"{'impl':>10} {'world':>5} {'size':>8} "
        f"{'init_alloc':>10} {'init_peak':>10} "
        f"{'prestep_alloc':>13} {'prestep_peak':>12} "
        f"{'poststep_alloc':>14} {'poststep_peak':>13}"
    )
    for row in rows:
        print(
            f"{row['impl']:>10} {row['world_size']:>5} {row['size']:>8} "
            f"{row['after_init_alloc_gib']:>10.3f} {row['after_init_peak_gib']:>10.3f} "
            f"{row['before_step_alloc_gib']:>13.3f} {row['before_step_peak_gib']:>12.3f} "
            f"{row['after_step_alloc_gib']:>14.3f} {row['after_step_peak_gib']:>13.3f}"
        )


def print_speed_table(
    rows: list[dict[str, Any]],
) -> None:    # 打印 optimizer state sharding 的训练步耗时与参数更新通信占比。
    print("========== Optimizer State Sharding Speed ==========")
    print(
        f"{'impl':>10} {'world':>5} {'size':>8} "
        f"{'step_ms':>10} {'std_ms':>9} {'param_comm_ms':>14} {'param_comm%':>12} {'tok/s':>12}"
    )
    for row in rows:
        print(
            f"{row['impl']:>10} {row['world_size']:>5} {row['size']:>8} "
            f"{row['avg_step_ms']:>10.3f} {row['std_step_ms']:>9.3f} "
            f"{row['param_update_comm_ms']:>14.3f} {row['param_update_comm_ratio'] * 100:>11.2f}% "
            f"{row['throughput_tok_s']:>12.1f}"
        )


def worker_main(
    args: argparse.Namespace,
) -> None:    # 单个分布式 worker：按 mode 跑 memory 或 speed，并在 rank 0 汇总打印。
    rank, world_size, _, device = init_process_group_from_env()

    if args.mode == "memory":
        spec = choose_spec(args.size)
        rows = [
            run_memory_profile(
                rank=rank,
                world_size=world_size,
                device=device,
                impl=impl,
                args=args,
                spec=spec,
            )
            for impl in args.impls
        ]
        if rank == 0:
            print_memory_table(rows)
    else:
        rows: list[dict[str, Any]] = []
        for size_name in args.sizes:
            spec = choose_spec(size_name)
            for impl in args.impls:
                rows.append(
                    run_speed_benchmark(
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        impl=impl,
                        args=args,
                        spec=spec,
                        size_name=size_name,
                    )
                )
        if rank == 0:
            print_speed_table(rows)

    dist.barrier()
    dist.destroy_process_group()


def local_spawn_entry(
    local_rank: int,
    world_size: int,
    master_port: int,
    args: argparse.Namespace,
) -> None:    # 单机模式下为每个子进程补齐 torchrun 风格环境变量，再复用同一套 worker。
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    worker_main(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="memory", choices=["memory", "speed"])
    parser.add_argument("--impls", nargs="+", default=["baseline", "sharded"], choices=["baseline", "sharded"])
    parser.add_argument("--size", type=str, default="xl", choices=list(SIZE_PRESETS.keys()))
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["small", "medium", "large", "xl", "2.7B"],
        choices=list(SIZE_PRESETS.keys()),
    )
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--local_batch_size", type=int, default=4)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="未使用 torchrun 时，单机自动 spawn 的进程数；默认取 min(2, 可见 GPU 数)。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("optimizer_state_sharding_benchmark 需要 CUDA 环境。")

    if launched_with_torchrun():
        worker_main(args)
        return

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count == 0:
        raise RuntimeError("没有可用的 CUDA 设备。")

    world_size = args.world_size if args.world_size is not None else min(2, visible_gpu_count)
    if world_size < 2:
        raise ValueError("optimizer state sharding benchmark 至少需要 2 个 rank。")

    master_port = find_free_port()
    mp.spawn(
        local_spawn_entry,
        args=(world_size, master_port, args),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
