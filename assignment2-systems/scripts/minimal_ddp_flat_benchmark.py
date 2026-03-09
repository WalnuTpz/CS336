"""
一个 minimal DDP flat benchmark 脚本：对比
1) 每个参数单独 all-reduce
2) 把所有梯度 flatten 后做一次 all-reduce

默认设置：
- 1 node x 2 GPUs
- model size = xl
- context_length = 256
- local_batch_size = 4
- dtype = fp16

用法示例：
  # 默认：同时对比 individual / flat
  uv run python scripts/minimal_ddp_flat_benchmark.py

  # 自定义步数
  uv run python scripts/minimal_ddp_flat_benchmark.py --warmup_steps 3 --steps 10

  # 只跑 flat 版本
  uv run python scripts/minimal_ddp_flat_benchmark.py --impls flat
"""

from __future__ import annotations

import argparse
import os
import socket
import timeit
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import DDPFlatGradients
from cs336_systems.ddp import DDPIndividualParameters


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


def find_free_port() -> int:    # 找一个本机可用端口，供本次进程组初始化使用。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def sync_if_cuda(device: torch.device) -> None:    # 只在 CUDA 上做同步，保证计时边界清晰。
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
    args: argparse.Namespace,
) -> ModelSpec:    # 根据 size preset 和命令行覆盖项确定最终模型规格。
    if args.size == "custom":
        missing = [name for name in ["d_model", "d_ff", "num_layers", "num_heads"] if getattr(args, name) is None]
        if missing:
            raise ValueError(f"size=custom 时必须提供这些参数：{missing}")
        return ModelSpec(
            d_model=args.d_model,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )

    base_spec = SIZE_PRESETS[args.size]
    return ModelSpec(
        d_model=args.d_model if args.d_model is not None else base_spec.d_model,
        d_ff=args.d_ff if args.d_ff is not None else base_spec.d_ff,
        num_layers=args.num_layers if args.num_layers is not None else base_spec.num_layers,
        num_heads=args.num_heads if args.num_heads is not None else base_spec.num_heads,
    )


def build_ddp_wrapper(
    impl: str,
    model: BasicsTransformerLM,
) -> torch.nn.Module:    # 按实现名选择 individual 或 flat 的最小 DDP 包装器。
    if impl == "individual":
        return DDPIndividualParameters(model)
    if impl == "flat":
        return DDPFlatGradients(model)
    raise ValueError(f"unknown impl: {impl}")


def benchmark_worker(
    rank: int,
    world_size: int,
    impl: str,
    master_addr: str,
    master_port: int,
    args: argparse.Namespace,
    spec: ModelSpec,
    result_queue: Any,
) -> None:    # 单个 rank 的 worker：跑一种 DDP 实现的 warm-up 和正式 benchmark。
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30),
    )

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    model = build_model(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        spec=spec,
    ).to(device=device, dtype=dtype)
    ddp_model = build_ddp_wrapper(
        impl,
        model,
    )
    optimizer = AdamW(ddp_model.parameters(), lr=args.lr)

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

        # 这里的 comm time 统计的是整个梯度同步阶段，flat 版本也包含 flatten/unflatten 的开销。
        sync_if_cuda(device)
        comm_t0 = timer()
        ddp_model.finish_gradient_synchronization()
        sync_if_cuda(device)
        comm_t1 = timer()

        optimizer.step()
        sync_if_cuda(device)
        t1 = timer()

        step_ms = (t1 - t0) * 1000.0
        comm_ms = (comm_t1 - comm_t0) * 1000.0
        return step_ms, comm_ms

    for _ in range(args.warmup_steps):
        run_one_step()

    step_times_ms: list[float] = []
    comm_times_ms: list[float] = []

    for _ in range(args.steps):
        local_step_ms, local_comm_ms = run_one_step()

        step_and_comm = torch.tensor(
            [local_step_ms, local_comm_ms],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(step_and_comm, op=dist.ReduceOp.MAX)

        if rank == 0:
            step_times_ms.append(float(step_and_comm[0].item()))
            comm_times_ms.append(float(step_and_comm[1].item()))

    if rank == 0:
        step_times = np.array(step_times_ms, dtype=np.float64)
        comm_times = np.array(comm_times_ms, dtype=np.float64)
        mean_step_ms = float(step_times.mean())
        std_step_ms = float(step_times.std(ddof=1)) if len(step_times) > 1 else 0.0
        mean_comm_ms = float(comm_times.mean())
        std_comm_ms = float(comm_times.std(ddof=1)) if len(comm_times) > 1 else 0.0
        comm_ratio = (mean_comm_ms / mean_step_ms) if mean_step_ms > 0 else 0.0
        global_tokens_per_step = world_size * local_batch_size * context_length
        throughput = global_tokens_per_step / (mean_step_ms / 1000.0)

        result_queue.put(
            {
                "impl": impl,
                "world_size": world_size,
                "dtype": args.dtype,
                "size": args.size,
                "d_model": spec.d_model,
                "d_ff": spec.d_ff,
                "num_layers": spec.num_layers,
                "num_heads": spec.num_heads,
                "local_batch_size": local_batch_size,
                "context_length": context_length,
                "avg_step_ms": mean_step_ms,
                "std_step_ms": std_step_ms,
                "avg_comm_ms": mean_comm_ms,
                "std_comm_ms": std_comm_ms,
                "comm_ratio": comm_ratio,
                "throughput": throughput,
                "step_times": step_times,
                "comm_times": comm_times,
            }
        )

    dist.barrier()
    dist.destroy_process_group()


def run_one_impl(
    impl: str,
    args: argparse.Namespace,
    spec: ModelSpec,
) -> dict[str, Any]:    # 跑一种 DDP 实现并返回 rank 0 汇总结果。
    ctx = mp.get_context("spawn")
    result_queue = ctx.SimpleQueue()
    master_port = find_free_port()

    mp.spawn(
        benchmark_worker,
        args=(
            args.world_size,
            impl,
            args.master_addr,
            master_port,
            args,
            spec,
            result_queue,
        ),
        nprocs=args.world_size,
        join=True,
    )

    if result_queue.empty():
        raise RuntimeError(f"{impl} benchmark did not return a result from rank 0")

    return result_queue.get()


def print_table(
    results: list[dict[str, Any]],
) -> None:    # 把不同实现的 step time / comm time 对比结果整理成表。
    print("\n========== Minimal DDP Flat Benchmark ==========")
    header = (
        f"{'impl':<12} {'step(ms)':<12} {'comm(ms)':<12} "
        f"{'comm(%)':<10} {'tok/s':<12}"
    )
    print(header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row['impl']:<12} {row['avg_step_ms']:<12.3f} {row['avg_comm_ms']:<12.3f} "
            f"{row['comm_ratio'] * 100:<10.2f} {row['throughput']:<12,.0f}"
        )

    print("================================================")


def print_comparison(
    results: list[dict[str, Any]],
) -> None:    # 在同时跑了两种实现时，额外打印一个简短对比结论。
    by_impl = {row["impl"]: row for row in results}
    if "individual" not in by_impl or "flat" not in by_impl:
        return

    individual = by_impl["individual"]
    flat = by_impl["flat"]

    step_speedup = individual["avg_step_ms"] / flat["avg_step_ms"]
    comm_speedup = individual["avg_comm_ms"] / flat["avg_comm_ms"]

    print(
        "Compare: "
        f"flat 的平均 step time 是 {flat['avg_step_ms']:.3f} ms，"
        f"individual 是 {individual['avg_step_ms']:.3f} ms，"
        f"step speedup = {step_speedup:.3f}x。"
    )
    print(
        "Compare: "
        f"flat 的平均通信阶段耗时是 {flat['avg_comm_ms']:.3f} ms，"
        f"individual 是 {individual['avg_comm_ms']:.3f} ms，"
        f"comm speedup = {comm_speedup:.3f}x。"
    )


def maybe_save_times(
    results: list[dict[str, Any]],
    times_out_dir: str | None,
) -> None:    # 若指定输出目录，则把每种实现的每步时间写成文本文件。
    if times_out_dir is None:
        return

    out_dir = Path(times_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in results:
        stacked = np.column_stack([row["step_times"], row["comm_times"]])
        out_path = out_dir / f"{row['impl']}_times.txt"
        np.savetxt(
            out_path,
            stacked,
            fmt="%.6f",
            header="step_ms comm_ms",
            comments="",
        )
        print(f"[info] times saved to {out_path}")


def main() -> None:    # 解析参数并在相同设置下对比 individual / flat 两种最小 DDP 实现。
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impls",
        nargs="+",
        default=["individual", "flat"],
        choices=["individual", "flat"],
    )
    parser.add_argument(
        "--size",
        type=str,
        default="xl",
        choices=list(SIZE_PRESETS.keys()) + ["custom"],
    )
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--local_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--times_out_dir", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("minimal_ddp_flat_benchmark 需要 CUDA 环境")

    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"需要至少 {args.world_size} 张 GPU，但当前只找到 {torch.cuda.device_count()} 张"
        )

    spec = choose_spec(args)
    results: list[dict[str, Any]] = []

    for impl in args.impls:
        results.append(
            run_one_impl(
                impl,
                args,
                spec,
            )
        )

    print_table(results)
    print_comparison(results)
    maybe_save_times(results, args.times_out_dir)


if __name__ == "__main__":
    main()
