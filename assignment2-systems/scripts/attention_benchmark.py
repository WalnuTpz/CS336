"""
一个 attention_benchmark 脚本：固定 batch size=4、移除 head 维度，
遍历 (impl, d_model, seq_len) 的组合，测 attention 的 forward / backward 耗时与显存。

- d_model ∈ [64, 256]
- seq_len ∈ [1024, 4096, 8192]
- impl ∈ [cs336, flash_pytorch, flash_triton]
- 每个配置：warm-up 后 forward 跑 iters 次计时；backward 跑 iters 次计时
- 每次 forward/backward 后都会 torch.cuda.synchronize()
- 默认参数偏向“差异明显但不过分耗时”的设置

用法示例：
  # bf16，默认测三种实现
  uv run python scripts/attention_benchmark.py --dtype bf16

  # 只测其中两种实现
  uv run python scripts/attention_benchmark.py --dtype bf16 --impls cs336 flash_triton

  # 自定义迭代次数 / warmup
  uv run python scripts/attention_benchmark.py --dtype fp32 --warmup 5 --iters 50
"""
import argparse
import traceback
import torch

from cs336_basics.model import scaled_dot_product_attention as cs336_sdp
from cs336_systems.flash_attention_pytorch import FlashAttentionForwardPytorch
from cs336_systems.flash_attention_triton import FlashAttentionForwardTriton


def get_attention_fn(which: str):
    if which == "cs336":
        def attention_fn(q, k, v):
            return cs336_sdp(Q=q, K=k, V=v, mask=None)

        return attention_fn

    if which == "flash_pytorch":
        def attention_fn(q, k, v):
            return FlashAttentionForwardPytorch.apply(q, k, v, False)

        return attention_fn

    if which == "flash_triton":
        def attention_fn(q, k, v):
            return FlashAttentionForwardTriton.apply(q, k, v, False)

        return attention_fn

    raise ValueError(f"unknown --impl: {which}")


@torch.no_grad()
def time_forward(attn, q, k, v, iters: int, warmup: int):
    # warm-up
    for _ in range(warmup):
        _ = attn(q, k, v)
        torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream()

    # 正式计时
    starter.record(stream)
    for _ in range(iters):
        _ = attn(q, k, v)
        torch.cuda.synchronize()
    ender.record(stream)
    torch.cuda.synchronize()

    ms = starter.elapsed_time(ender)
    return ms / iters


def time_backward(attn, q, k, v, iters: int, warmup: int):
    # warm-up（带 backward）
    for _ in range(warmup):
        out = attn(q, k, v)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        q.grad = None
        k.grad = None
        v.grad = None

    # 正式计时
    torch.cuda.reset_peak_memory_stats()

    mem_before_bw_bytes = None

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream()

    starter.record(stream)
    for i in range(iters):
        out = attn(q, k, v)
        torch.cuda.synchronize()

        if i == 0:
            mem_before_bw_bytes = torch.cuda.memory_allocated()

        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()

        q.grad = None
        k.grad = None
        v.grad = None
    ender.record(stream)
    torch.cuda.synchronize()

    ms = starter.elapsed_time(ender)
    peak_bytes = torch.cuda.max_memory_allocated()
    return ms / iters, mem_before_bw_bytes, peak_bytes


def bytes_to_gib(x: int) -> float:
    return x / (1024 ** 3)


def _format_metric(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument(
        "--impls",
        nargs="+",
        default=["cs336", "flash_pytorch", "flash_triton"],
        choices=["cs336", "flash_pytorch", "flash_triton"],
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--compile", action="store_true", help="使用 torch.compile 将 attention 进行编译")
    p.add_argument("--compile_mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--fullgraph", action="store_true")

    args = p.parse_args()

    assert args.device == "cuda", "本题要求 cuda benchmark"
    assert torch.cuda.is_available(), "no CUDA available"

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    B = 4
    d_models = [64, 256]
    seq_lens = [1024, 4096, 8192]

    results = []

    print("\n====== Benchmark results ======")
    print(
        f"{'impl':>14} {'d_model':>6} {'seq_len':>7} {'status':>6} {'fwd_ms':>10} {'bwd_ms':>10} {'memBW(GiB)':>11} {'peak(GiB)':>10}",
        flush=True,
    )

    for impl in args.impls:
        attn = get_attention_fn(impl)
        if args.compile:  # 触发编译
            attn = torch.compile(attn, mode=args.compile_mode, fullgraph=args.fullgraph)

        for d in d_models:
            for L in seq_lens:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                row = {
                    "impl": impl,
                    "d_model": d,
                    "seq_len": L,
                    "status": "ok",
                    "fwd_ms": None,
                    "bwd_ms": None,
                    "mem_before_bw_gib": None,
                    "peak_gib": None,
                }

                try:
                    # 随机输入 Q,K,V，去掉 head 维度：shape [B, L, D]
                    q = torch.randn(B, L, d, device="cuda", dtype=dtype, requires_grad=True)
                    k = torch.randn(B, L, d, device="cuda", dtype=dtype, requires_grad=True)
                    v = torch.randn(B, L, d, device="cuda", dtype=dtype, requires_grad=True)

                    if args.compile:    # 触发编译（不要把编译时间算进计时）
                        _ = attn(q, k, v)
                        torch.cuda.synchronize()

                    row["fwd_ms"] = time_forward(attn, q, k, v, iters=args.iters, warmup=args.warmup)

                    bwd_ms, mem_before_bw, peak = time_backward(
                        attn, q, k, v, iters=args.iters, warmup=max(3, args.warmup // 2)
                    )
                    row["bwd_ms"] = bwd_ms
                    row["mem_before_bw_gib"] = bytes_to_gib(mem_before_bw)
                    row["peak_gib"] = bytes_to_gib(peak)

                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg:
                        row["status"] = "oom"
                        # 清理 OOM 后的状态
                        del q, k, v
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    else:
                        row["status"] = f"error: {type(e).__name__}"
                        print("ERROR:", e)
                        traceback.print_exc()

                results.append(row)
                print(
                    f"{row['impl']:>14} {row['d_model']:>6} {row['seq_len']:>7} {row['status']:>6} "
                    f"{_format_metric(row['fwd_ms']):>10} { _format_metric(row['bwd_ms']):>10} "
                    f"{_format_metric(row['mem_before_bw_gib']):>11} { _format_metric(row['peak_gib']):>10}",
                    flush=True,
                )
    print("\n================================")

if __name__ == "__main__":
    main()
