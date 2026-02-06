"""
一个最基础的端到端 benchmark 脚本：测 forward 或 forward+backward 的耗时。

用法示例：
  # forward only
  uv run python scripts/benchmarking.py --size small --context_length 256 --warmup_steps 5 --steps 50

  # forward + backward
  uv run python scripts/benchmarking.py --size small --context_length 256 --warmup_steps 5 --steps 50 --backward

  # 自定义超参数
  uv run python scripts/benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --context_length 256 --backward
"""

from __future__ import annotations

import argparse
import inspect
import timeit
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------
# 1) 模型 size 预设（来自文档 Table 1）
# -----------------------------
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


# -----------------------------
# 2) 选择并构造 basics Transformer 模型
# -----------------------------
def _pick_transformer_class(module: Any) -> type:
    """
    在 cs336_basics.model 里尽量找到“Transformer 语言模型”对应的类名。
    为了兼容不同实现，这里做一些容错：
      - 优先尝试常见类名
      - 找不到再退化为“名字里包含 Transformer 的第一个类”
    """
    preferred_names = [
        "TransformerLM",
        "TransformerLanguageModel",
        "TransformerModel",
        "Transformer",
    ]
    for name in preferred_names:
        if hasattr(module, name) and inspect.isclass(getattr(module, name)):
            return getattr(module, name)

    # 退化：找名字包含 Transformer 的类
    for name, obj in vars(module).items():
        if inspect.isclass(obj) and "Transformer" in name:
            return obj

    raise RuntimeError(
        "在 cs336_basics.model 中没有找到 Transformer 相关的模型类。"
        "请检查你的 cs336_basics.model 里模型类的名字。"
    )


def _build_model_from_signature(
    ModelCls: type,
    *,
    vocab_size: int,
    context_length: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    rope_theta: float,
) -> Any:
    """
    通过检查 __init__ 的参数名，把我们常见的超参数“对齐”进去。
    这样可以尽量兼容不同同学/不同版本实现的构造函数参数名。
    """
    sig = inspect.signature(ModelCls.__init__)
    params = sig.parameters

    # 可能出现的“同义参数名”映射
    def pick(name_candidates: Tuple[str, ...], value: Any, out: Dict[str, Any]):
        for n in name_candidates:
            if n in params:
                out[n] = value
                return

    kwargs: Dict[str, Any] = {}

    pick(("vocab_size", "n_vocab", "vocab"), vocab_size, kwargs)
    pick(("context_length", "max_seq_len", "seq_len", "block_size"), context_length, kwargs)
    pick(("d_model", "dim", "n_embd", "embed_dim"), d_model, kwargs)
    pick(("d_ff", "ffn_dim", "mlp_dim", "d_hidden"), d_ff, kwargs)
    pick(("num_layers", "n_layers", "n_layer"), num_layers, kwargs)
    pick(("num_heads", "n_heads", "n_head"), num_heads, kwargs)

    # RoPE theta：有的实现会要
    pick(("rope_theta", "theta"), rope_theta, kwargs)

    # 有些实现可能要求额外参数（例如 dropout），这里不强行传，避免不匹配
    try:
        model = ModelCls(**kwargs)
    except TypeError as e:
        # 如果你自己的实现构造函数参数名非常不一样，这里会报错
        raise TypeError(
            f"构造模型失败：{e}\n"
            f"尝试传入的参数为：{kwargs}\n"
            "请打开 cs336_basics/model.py 看看你的模型 __init__ 需要哪些参数名，"
            "然后在 _build_model_from_signature 里补充同义参数名映射。"
        )
    return model


# -----------------------------
# 3) 单步 forward / forward+backward
# -----------------------------
def _run_forward_only(model: Any, x: torch.Tensor) -> torch.Tensor:
    """
    只跑 forward：返回 logits（或 forward 的输出）。
    """
    out = model(x)
    # 有的模型可能返回 (logits, extra...) 或者一个带 logits 属性的对象
    if isinstance(out, tuple) and len(out) > 0:
        return out[0]
    if hasattr(out, "logits"):
        return out.logits
    return out


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


# -----------------------------
# 4) Benchmark 主流程
# -----------------------------
def benchmark(
    *,
    model: Any,
    device: torch.device,
    x: torch.Tensor,
    y: torch.Tensor,
    warmup_steps: int,
    steps: int,
    do_backward: bool,
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
    for _ in range(warmup_steps):
        if do_backward:
            model.zero_grad(set_to_none=True)
            logits = _run_forward_only(model, x)
            loss = _compute_loss_from_logits(logits, y)
            loss.backward()
        else:
            with torch.no_grad():
                _ = _run_forward_only(model, x)
        _sync_if_cuda(device)

    # 正式计时：记录每一步耗时
    times = []
    timer = timeit.default_timer

    for _ in range(steps):
        t0 = timer()

        if do_backward:
            model.zero_grad(set_to_none=True)
            logits = _run_forward_only(model, x)
            loss = _compute_loss_from_logits(logits, y)
            loss.backward()
        else:
            with torch.no_grad():
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

    # 运行控制
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--backward", action="store_true", help="若指定则跑 forward+backward；否则只跑 forward")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # 设备选择
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] 你指定了 --device cuda 但当前环境没有 CUDA，自动切到 CPU。")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # dtype 选择（注意：输入 token 是 int64，不受 dtype 影响；dtype 主要影响模型参数）
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

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

    repo_root = Path(__file__).resolve().parents[1]
    basics_dir = repo_root / "cs336-basics"
    if str(basics_dir) not in sys.path:
        sys.path.insert(0, str(basics_dir))
    # 导入 basics 模型
    import cs336_basics.model as basics_model

    ModelCls = _pick_transformer_class(basics_model)
    model = _build_model_from_signature(
        ModelCls,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=spec.d_model,
        d_ff=spec.d_ff,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        rope_theta=args.rope_theta,
    )

    # 把模型放到设备上，并设置参数 dtype（CPU 上用 fp16/bf16 可能很慢/不支持，按需自行改）
    model = model.to(device=device)
    if device.type == "cuda":
        model = model.to(dtype=dtype)

    # 随机 batch（token ids）
    # x: [B, T]，y: [B, T]（这里用 x 的 shift 版本更像语言模型训练）
    B = args.batch_size
    T = args.context_length
    V = args.vocab_size

    x = torch.randint(low=0, high=V, size=(B, T), device=device, dtype=torch.long)
    y = torch.roll(x, shifts=-1, dims=1)

    # 为了公平：先同步一下，避免把之前的 CUDA 工作混进来
    _sync_if_cuda(device)

    mean_ms, std_ms, times_ms = benchmark(
        model=model,
        device=device,
        x=x,
        y=y,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        do_backward=args.backward,
    )

    mode = "forward+backward" if args.backward else "forward-only"
    tokens_per_step = B * T
    # 吞吐量：tokens/s（简单估算）
    tok_per_s = tokens_per_step / (mean_ms / 1000.0)

    print("========== Benchmark Result ==========")
    print(f"mode          : {mode}")
    print(f"device        : {device}")
    print(f"dtype         : {dtype} (仅影响模型参数；输入 token 仍为 int64)")
    print(f"size          : {args.size}")
    print(f"spec          : d_model={spec.d_model}, d_ff={spec.d_ff}, num_layers={spec.num_layers}, num_heads={spec.num_heads}")
    print(f"batch/context : B={B}, T={T}, vocab_size={V}")
    print(f"warmup/steps  : warmup={args.warmup_steps}, steps={args.steps}")
    print("--------------------------------------")
    print(f"avg step time : {mean_ms:.3f} ms")
    print(f"std step time : {std_ms:.3f} ms")
    print(f"throughput    : {tok_per_s:,.0f} tokens/s")
    print("======================================")

    # 如果你想把每步数据导出到文件，方便做表格/画图，可以在这里写 np.savetxt / pandas


if __name__ == "__main__":
    main()
