"""
一个单机多进程 all-reduce benchmark 脚本。

默认扫描：
- backend + device：
    1) gloo + cpu
    2) nccl + gpu
- world_size ∈ [2, 4, 6]
- tensor_size ∈ [1, 10, 100, 1024] MB
- dtype = float32

输出：
- 终端汇总表

用法示例：
  # 使用默认扫描空间
  uv run python scripts/all_reduce_benchmark.py

  # 只测 NCCL
  uv run python scripts/all_reduce_benchmark.py --backends nccl

  # 自定义 world size / tensor size
  uv run python scripts/all_reduce_benchmark.py --world_sizes 2 4 --sizes_mb 1 10 100
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


DEFAULT_SIZES_MB = [1, 10, 100, 1024]
DEFAULT_WORLD_SIZES = [2, 4, 6]
DEFAULT_BACKENDS = ["gloo", "nccl"]


def find_free_port() -> int:    # 找一个本机可用端口，供本次进程组初始化使用。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def size_mb_to_numel(
    size_mb: int,
    dtype: torch.dtype = torch.float32,
) -> int:    # 把张量大小从 MB 换算成对应 dtype 下的元素个数。
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    size_bytes = size_mb * 1024 * 1024
    return size_bytes // bytes_per_elem


def pick_warmup_and_iters(
    backend: str,
    size_mb: int,
) -> tuple[int, int]:    # 按 backend 和张量大小选择更合适的 warm-up / iters。
    if backend == "gloo":
        if size_mb <= 1:
            return 5, 30
        if size_mb <= 10:
            return 5, 20
        if size_mb <= 100:
            return 2, 8
        return 1, 2

    if size_mb <= 1:
        return 10, 100
    if size_mb <= 10:
        return 5, 50
    if size_mb <= 100:
        return 3, 20
    return 2, 8


def sync_if_needed(device: torch.device) -> None:    # 只在 CUDA 上做同步，保证计时边界清晰。
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_device(
    backend: str,
    rank: int,
) -> torch.device:    # 按 backend 选择当前 rank 应该绑定的设备。
    if backend == "nccl":
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def bench_worker(
    rank: int,
    world_size: int,
    backend: str,
    size_mb: int,
    master_addr: str,
    master_port: int,
    timeout_seconds: int,
    result_queue: Any,
) -> None:    # 单个 rank 的 worker：初始化进程组，执行 warm-up 和正式计时。
    # 每个子进程都用同一组 rendezvous 参数加入进程组。
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    device = get_device(backend, rank)
    if backend == "gloo":
        # 避免多进程下 CPU 线程过度竞争
        torch.set_num_threads(1)

    try:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_seconds),
        )

        dtype = torch.float32
        numel = size_mb_to_numel(size_mb, dtype=dtype)
        x = torch.randn(numel, device=device, dtype=dtype)

        warmup, iters = pick_warmup_and_iters(backend, size_mb)

        # warm-up：不把首次通信的初始化成本记入正式计时
        for _ in range(warmup):
            dist.barrier()
            sync_if_needed(device)
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            sync_if_needed(device)

        # 正式计时：每次都在 collective 前后同步，避免把前一轮工作混进来。
        local_times_ms: list[float] = []
        for _ in range(iters):
            dist.barrier()
            sync_if_needed(device)

            t0 = time.perf_counter()
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            sync_if_needed(device)
            t1 = time.perf_counter()

            local_times_ms.append((t1 - t0) * 1000.0)

        local_stats = {
            "rank": rank,
            "mean_ms": sum(local_times_ms) / len(local_times_ms),
            "min_ms": min(local_times_ms),
            "max_ms": max(local_times_ms),
            "iters": iters,
            "warmup": warmup,
        }

        gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_stats)

        if rank == 0:
            # all-reduce 的整体延迟取最慢 rank，更符合 collective 的完成时间。
            valid_stats = [item for item in gathered if item is not None]
            latency_ms = max(item["mean_ms"] for item in valid_stats)
            min_rank_ms = min(item["mean_ms"] for item in valid_stats)
            max_rank_ms = max(item["mean_ms"] for item in valid_stats)

            # 用单次 all-reduce 的 payload 大小估算有效带宽，单位统一成 GiB/s。
            size_bytes = size_mb * 1024 * 1024
            bandwidth_gibps = (size_bytes / (1024 ** 3)) / (latency_ms / 1000.0)

            result_queue.put(
                {
                    "status": "ok",
                    "backend": backend,
                    "device": device.type,
                    "world_size": world_size,
                    "size_mb": size_mb,
                    "dtype": "float32",
                    "latency_ms": latency_ms,
                    "min_rank_ms": min_rank_ms,
                    "max_rank_ms": max_rank_ms,
                    "bandwidth_gibps": bandwidth_gibps,
                    "iters": iters,
                    "warmup": warmup,
                    "reason": None,
                    "error": None,
                }
            )

    except Exception as e:
        if rank == 0:
            result_queue.put(
                {
                    "status": "error",
                    "backend": backend,
                    "device": device.type,
                    "world_size": world_size,
                    "size_mb": size_mb,
                    "dtype": "float32",
                    "latency_ms": None,
                    "min_rank_ms": None,
                    "max_rank_ms": None,
                    "bandwidth_gibps": None,
                    "iters": None,
                    "warmup": None,
                    "reason": None,
                    "error": repr(e),
                }
            )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def run_one_config(
    *,
    backend: str,
    world_size: int,
    size_mb: int,
    master_addr: str,
    timeout_seconds: int,
) -> dict[str, Any]:    # 跑一个 (backend, world_size, size_mb) 配置并返回结果。
    if backend == "nccl":
        n_gpus = torch.cuda.device_count()
        if n_gpus < world_size:
            return {
                "status": "skipped",
                "backend": backend,
                "device": "cuda",
                "world_size": world_size,
                "size_mb": size_mb,
                "dtype": "float32",
                "latency_ms": None,
                "min_rank_ms": None,
                "max_rank_ms": None,
                "bandwidth_gibps": None,
                "iters": None,
                "warmup": None,
                "reason": f"Need {world_size} GPUs, but only found {n_gpus}.",
                "error": None,
            }

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    master_port = find_free_port()

    try:
        # 一个配置对应一次独立的多进程 benchmark。
        mp.spawn(
            bench_worker,
            args=(
                world_size,
                backend,
                size_mb,
                master_addr,
                master_port,
                timeout_seconds,
                result_queue,
            ),
            nprocs=world_size,
            join=True,
        )
    except Exception as e:
        return {
            "status": "error",
            "backend": backend,
            "device": "cuda" if backend == "nccl" else "cpu",
            "world_size": world_size,
            "size_mb": size_mb,
            "dtype": "float32",
            "latency_ms": None,
            "min_rank_ms": None,
            "max_rank_ms": None,
            "bandwidth_gibps": None,
            "iters": None,
            "warmup": None,
            "reason": None,
            "error": f"mp.spawn failed: {repr(e)}",
        }

    if result_queue.empty():
        return {
            "status": "error",
            "backend": backend,
            "device": "cuda" if backend == "nccl" else "cpu",
            "world_size": world_size,
            "size_mb": size_mb,
            "dtype": "float32",
            "latency_ms": None,
            "min_rank_ms": None,
            "max_rank_ms": None,
            "bandwidth_gibps": None,
            "iters": None,
            "warmup": None,
            "reason": None,
            "error": "No result returned from rank 0.",
        }

    return result_queue.get()


def print_table(results: list[dict[str, Any]]) -> None:    # 把所有结果整理成终端汇总表。
    ok_rows = [row for row in results if row["status"] == "ok"]
    skip_rows = [row for row in results if row["status"] == "skipped"]
    err_rows = [row for row in results if row["status"] == "error"]

    # 先按 backend / world_size / size 排序，汇总表更容易横向比较。
    ok_rows.sort(key=lambda row: (row["backend"], row["world_size"], row["size_mb"]))

    header = (
        f"{'backend':<8} {'device':<6} {'world':<5} {'size(MB)':<8} "
        f"{'latency(ms)':<12} {'GiB/s':<10} {'iters':<5}"
    )
    print("\n=== All-Reduce Benchmark Results ===")
    print(header)
    print("-" * len(header))

    for row in ok_rows:
        print(
            f"{row['backend']:<8} {row['device']:<6} {row['world_size']:<5} {row['size_mb']:<8} "
            f"{row['latency_ms']:<12.3f} {row['bandwidth_gibps']:<10.3f} {row['iters']:<5}"
        )

    if skip_rows:
        print("\n=== Skipped ===")
        for row in skip_rows:
            print(
                f"{row['backend']} world_size={row['world_size']} size={row['size_mb']}MB -> {row['reason']}"
            )

    if err_rows:
        print("\n=== Errors ===")
        for row in err_rows:
            print(
                f"{row['backend']} world_size={row['world_size']} size={row['size_mb']}MB -> {row['error']}"
            )


def main() -> None:    # 解析参数，逐个配置执行 benchmark，并在最后统一输出。
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--timeout_seconds", type=int, default=600)
    parser.add_argument("--sizes_mb", nargs="+", type=int, default=DEFAULT_SIZES_MB)
    parser.add_argument("--world_sizes", nargs="+", type=int, default=DEFAULT_WORLD_SIZES)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        choices=["gloo", "nccl"],
    )
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    results: list[dict[str, Any]] = []

    # 逐个配置执行，最后统一打印汇总表。
    for backend in args.backends:
        for world_size in args.world_sizes:
            for size_mb in args.sizes_mb:
                row = run_one_config(
                    backend=backend,
                    world_size=world_size,
                    size_mb=size_mb,
                    master_addr=args.master_addr,
                    timeout_seconds=args.timeout_seconds,
                )
                results.append(row)

    print_table(results)


if __name__ == "__main__":
    main()
