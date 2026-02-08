"""
一个最基础的端到端 benchmark 脚本：测 forward 或 forward+backward 的耗时。

用法示例：
  # forward only
  uv run python scripts/benchmark.py --size small --context_length 256 --warmup_steps 5 --steps 50

  # forward + backward
  uv run python scripts/benchmark.py --size small --context_length 256 --warmup_steps 5 --steps 50 --backward

  # 自定义超参数
  uv run python scripts/benchmark.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --context_length 256 --backward
"""

from __future__ import annotations

import argparse
import timeit
from dataclasses import dataclass
from typing import Dict, Tuple
from contextlib import contextmanager, nullcontext
import torch.cuda.nvtx as nvtx
import math
import numpy as np
import torch
import torch.nn.functional as F

import cs336_basics.model as model_mod
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


# NVTX 标注，用于 Nsight Systems 里按 range 过滤/统计
@contextmanager
def nvtx_range(name: str, enabled: bool):
    """NVTX range：enabled=False 或 CPU 时自动空操作。"""
    if enabled and torch.cuda.is_available():
        nvtx.range_push(name)
        try:
            yield
        finally:
            nvtx.range_pop()
    else:
        yield


# 模型 size 预设
@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


SIZE_PRESETS: Dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelSpec(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelSpec(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": ModelSpec(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


# 构造 basics Transformer 模型
def _build_model(
    *,
    vocab_size: int,
    context_length: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    rope_theta: float,
):
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )


# 单步 forward / forward+backward
def _run_forward_only(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """只跑 forward，直接返回 logits。"""
    return model(x)


def _compute_loss_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    用 logits 计算一个标量 loss，便于 backward。
    默认假设 logits 形状为 [B, T, V] 或 [B, V]。
    """
    if logits.dim() == 3:
        b, t, v = logits.shape
        return F.cross_entropy(logits.reshape(b * t, v), y.reshape(b * t))
    if logits.dim() == 2:
        b, v = logits.shape
        return F.cross_entropy(logits.reshape(b, v), y.reshape(b))
    raise RuntimeError(f"不支持的 logits 维度：{logits.shape}")


def _sync_if_cuda(device: torch.device) -> None:
    """
    CUDA 上同步，CPU 上不做事。
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


# 带 NVTX range 的 scaled_dot_product_attention
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    # 1) QK^T matmul
    with nvtx_range("attn/qk_matmul", enabled=True):
        attention_scores = model_mod.einsum(
            Q, K, "... query d_k, ... key d_k -> ... query key"
        ) / math.sqrt(d_k)

    # 2) mask（可选，但通常很快；放着不影响）
    if mask is not None:
        with nvtx_range("attn/mask", enabled=True):
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    # 3) softmax
    with nvtx_range("attn/softmax", enabled=True):
        attention_weights = model_mod.softmax(attention_scores, dim=-1)

    # 4) PV matmul
    with nvtx_range("attn/pv_matmul", enabled=True):
        out = model_mod.einsum(
            attention_weights, V, "... query key, ... key d_v -> ... query d_v"
        )
    return out


# Benchmark 主流程
def benchmark(
    *,
    model: torch.nn.Module,
    optimizer: torch.nn.Module | None,
    device: torch.device,
    x: torch.Tensor,
    y: torch.Tensor,
    warmup_steps: int,
    steps: int,
    do_backward: bool,
    do_optimizer_step: bool,
    nvtx_enabled: bool,
    amp_ctx: bool
) -> Tuple[float, float, np.ndarray]:
    """
    返回：
      - mean_ms: 每步平均耗时（毫秒）
      - std_ms:  每步耗时标准差（毫秒）
      - times_ms: 每步耗时数组（毫秒）
    """
    # forward-only 建议 eval + no_grad；backward 需要 train + grad
    if do_backward:
        model.train()
    else:
        model.eval()

    # 预热：不计时
    with nvtx_range("warm_up", enabled=nvtx_enabled):
        for _ in range(warmup_steps):
            if do_backward:
                model.zero_grad(set_to_none=True)

                with amp_ctx:
                    logits = _run_forward_only(model, x)
                loss = _compute_loss_from_logits(logits.float(), y)

                loss.backward()

                if do_optimizer_step:
                    assert optimizer is not None
                    optimizer.step()
            else:
                with torch.no_grad():
                    with amp_ctx:
                        _ = _run_forward_only(model, x)
            _sync_if_cuda(device)

    # 正式计时：记录每一步耗时
    times = []
    timer = timeit.default_timer

    for _ in range(steps):
        t0 = timer()

        with nvtx_range("PROFILE_STEP", enabled=nvtx_enabled):
            if do_backward:
                model.zero_grad(set_to_none=True)

                with nvtx_range("forward", enabled=nvtx_enabled):
                    with amp_ctx:
                        logits = _run_forward_only(model, x)
                    loss = _compute_loss_from_logits(logits.float(), y)

                with nvtx_range("backward", enabled=nvtx_enabled):
                    loss.backward()

                if do_optimizer_step:
                    assert optimizer is not None
                    with nvtx_range("optimizer", enabled=nvtx_enabled):
                        optimizer.step()
            else:
                with torch.no_grad():
                    with nvtx_range("forward", enabled=nvtx_enabled):
                        with amp_ctx:
                            _ = _run_forward_only(model, x)

        # 题目要求：每一步后 synchronize
        _sync_if_cuda(device)

        t1 = timer()
        times.append((t1 - t0) * 1000.0)  # 转成 ms

    times_ms = np.array(times, dtype=np.float64)
    mean_ms = float(times_ms.mean())
    std_ms = float(times_ms.std(ddof=1)) if len(times_ms) > 1 else 0.0
    return mean_ms, std_ms, times_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    # 模型预设 size（可选）
    parser.add_argument("--size", type=str, default="small", choices=list(SIZE_PRESETS.keys()) + ["custom"],
                        help="模型大小预设；custom 表示完全用手动参数")
    # 手动超参数（size=custom 时必须给齐；否则可用于覆盖 preset）
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)

    # 通用超参数
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--times_out", type=str, default=None, help="若提供，则把每步耗时(ms)写入该文件")
    parser.add_argument("--lr", type=float, default=1e-3)

    # 运行控制
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--backward", action="store_true", help="若指定则跑 forward+backward；否则只跑 forward")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--amp_bf16", action="store_true", help = "启用自动混合精度，让部分计算用 bfloat16 来跑，但模型权重仍保持 fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer_step", action="store_true", help="跑 optimizer.step()（完整训练一步）")
    parser.add_argument("--nvtx", action="store_true", help="开启 NVTX ranges（用于 nsys 过滤/统计）")

    args = parser.parse_args()

    if args.optimizer_step and not args.backward:
        raise ValueError("--optimizer_step 需要配合 --backward（完整训练一步：forward+backward+optimizer）")

    # 设备选择
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] 你指定了 --device cuda 但当前环境没有 CUDA，自动切到 CPU。")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # dtype 选择（注意：输入 token 是 int64，不受 dtype 影响；dtype 主要影响模型参数）
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # 是否启用混合精度
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (args.amp_bf16 and device.type == "cuda")
        else nullcontext()
    )

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # 读取 size preset，并允许命令行覆盖
    if args.size == "custom":
        missing = [k for k in ["d_model", "d_ff", "num_layers", "num_heads"] if getattr(args, k) is None]
        if missing:
            raise ValueError(f"size=custom 时必须提供这些参数：{missing}")
        spec = ModelSpec(args.d_model, args.d_ff, args.num_layers, args.num_heads)
    else:
        spec = SIZE_PRESETS[args.size]
        # 允许覆盖 preset
        d_model = args.d_model if args.d_model is not None else spec.d_model
        d_ff = args.d_ff if args.d_ff is not None else spec.d_ff
        num_layers = args.num_layers if args.num_layers is not None else spec.num_layers
        num_heads = args.num_heads if args.num_heads is not None else spec.num_heads
        spec = ModelSpec(d_model=d_model, d_ff=d_ff, num_layers=num_layers, num_heads=num_heads)

    # 创建模型
    _orig_sdp = model_mod.scaled_dot_product_attention
    if args.nvtx:
        model_mod.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    model = _build_model(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=spec.d_model,
        d_ff=spec.d_ff,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        rope_theta=args.rope_theta,
    )

    # 把模型放到设备上，并设置参数 dtype
    model = model.to(device=device)
    if device.type == "cuda" and not args.amp_bf16:
        model = model.to(dtype=dtype)

    optimizer = None
    if args.optimizer_step:
        optimizer = AdamW(model.parameters(), lr=args.lr)

    # 随机 batch（token ids）
    # x: [B, T]，y: [B, T]
    B = args.batch_size
    T = args.context_length
    V = args.vocab_size

    x = torch.randint(low=0, high=V, size=(B, T), device=device, dtype=torch.long)
    y = torch.roll(x, shifts=-1, dims=1)

    # 先进行一次同步，避免把之前的 CUDA 工作混进来
    _sync_if_cuda(device)

    mean_ms, std_ms, times_ms = benchmark(
        model=model,
        optimizer=optimizer,
        device=device,
        x=x,
        y=y,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        do_backward=args.backward,
        do_optimizer_step=args.optimizer_step,
        nvtx_enabled=args.nvtx,
        amp_ctx=amp_context
    )

    mode = "forward+backward" if args.backward else "forward-only"
    tokens_per_step = B * T
    # 吞吐量：tokens/s
    tok_per_s = tokens_per_step / (mean_ms / 1000.0)

    print("========== Benchmark Result ==========")
    print(f"mode          : {mode}")
    print(f"device        : {device}")
    print(f"dtype         : {dtype}")
    print(f"amp_bf16      : {args.amp_bf16}")
    print(f"size          : {args.size}")
    print(f"spec          : d_model={spec.d_model}, d_ff={spec.d_ff}, num_layers={spec.num_layers}, num_heads={spec.num_heads}")
    print(f"batch/context : B={B}, T={T}, vocab_size={V}")
    print(f"warmup/steps  : warmup={args.warmup_steps}, steps={args.steps}")
    print("--------------------------------------")
    print(f"avg step time : {mean_ms:.3f} ms")
    print(f"std step time : {std_ms:.3f} ms")
    print(f"throughput    : {tok_per_s:,.0f} tokens/s")
    print("======================================")

    # 若指定 times_out，则导出每步耗时，便于后续画图/统计
    if args.times_out:
        np.savetxt(args.times_out, times_ms, fmt="%.6f")
        print(f"[info] times saved to {args.times_out}")


if __name__ == "__main__":
    main()
