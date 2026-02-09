"""
一个 attention_benchmark 脚本：固定 batch size=8、移除 head 维度，
遍历 (d_model, seq_len) 的组合，测 attention 的 forward / backward 耗时与显存。

- d_model ∈ [16, 32, 64, 128]
- seq_len ∈ [256, 1024, 4096, 8192, 16384]
- 每个配置：warm-up 后 forward 跑 iters 次计时；backward 跑 iters 次计时
- 每次 forward/backward 后都会 torch.cuda.synchronize()

用法示例：
  # bf16，默认 iters=100, warmup=10
  uv run python scripts/pytorch_attention_benchmark.py --dtype bf16

  # 自定义迭代次数 / warmup
  uv run python scripts/pytorch_attention_benchmark.py --dtype bf16 --warmup 5 --iters 50

  # fp32（更慢但更稳）
  uv run python scripts/pytorch_attention_benchmark.py --dtype fp32 --warmup 5 --iters 50
"""
import argparse
import traceback
import torch
import inspect

from cs336_basics.model import scaled_dot_product_attention as cs336_sdp


def get_attention_fn(which: str):
    if which != "cs336":
        raise ValueError("only --impl cs336 is supported in this script")

    params = inspect.signature(cs336_sdp).parameters

    def attention_fn(q, k, v):
        kwargs = {}
        if "is_causal" in params:
            kwargs["is_causal"] = False
        if "attn_mask" in params:
            kwargs["attn_mask"] = None
        if "mask" in params:
            kwargs["mask"] = None
        if "dropout_p" in params:
            kwargs["dropout_p"] = 0.0
        return cs336_sdp(q, k, v, **kwargs)

    return attention_fn


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--impl", default="cs336", choices=["cs336"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
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

    attn = get_attention_fn(args.impl)
    if args.compile:  # 触发编译
        attn = torch.compile(attn, mode=args.compile_mode, fullgraph=args.fullgraph)

    B = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]

    results = []

    for d in d_models:
        for L in seq_lens:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            row = {
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

                # 进行 100 次 forward 计时
                # noinspection PyTypeChecker
                row["fwd_ms"] = time_forward(attn, q, k, v, iters=args.iters, warmup=args.warmup)

                # backward 之前测显存 + 进行 100 次 backward 计时
                bwd_ms, mem_before_bw, peak = time_backward(attn, q, k, v, iters=args.iters, warmup=max(3, args.warmup // 2))
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

    print("\n====== Benchmark results ======")
    print(
        f"{'d_model':>6} {'seq_len':>7} {'status':>6} {'fwd_ms':>10} {'bwd_ms':>10} {'memBW(GiB)':>11} {'peak(GiB)':>10}")
    for r in results:
        def fmt(x, nd=3):
            if x is None:
                return "-"
            if isinstance(x, float):
                return f"{x:.{nd}f}"
            return str(x)

        print(
            f"{r['d_model']:>6} {r['seq_len']:>7} {r['status']:>6} "
            f"{fmt(r['fwd_ms']):>10} {fmt(r['bwd_ms']):>10} "
            f"{fmt(r['mem_before_bw_gib']):>11} {fmt(r['peak_gib']):>10}"
        )
    print("\n================================")

if __name__ == "__main__":
    main()
