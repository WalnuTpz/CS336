from __future__ import annotations

import argparse
import numpy as np
import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.generation import generate  


def _dtype_from_str(dtype_str: str) -> torch.dtype:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_str}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS336 Assignment1: Generate text from a trained TransformerLM")

    # 路径
    parser.add_argument("--ckpt", type=str, required=True, help="Path to ckpt_*.pt saved by training script.")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab.json used by Tokenizer.from_files.")
    parser.add_argument("--merges", type=str, required=True, help="Path to merges.json used by Tokenizer.from_files.")

    # 模型超参数（必须与训练一致，否则 load_state_dict 会 shape mismatch）
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

    # 采样参数
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)

    # 其他
    parser.add_argument("--device", type=str, default=None, help="e.g. cpu, cuda, cuda:0. Default auto-detect.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eos_token", type=str, default="<|endoftext|>", help="Optional early-stop token (truncate output).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 自动选择 device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    device_t = torch.device(device)

    # 随机种子（影响 multinomial）
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # dtype（CPU 上用半精度常常不划算/不支持）
    dtype = _dtype_from_str(args.dtype)
    if device_t.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        print(f"[warn] device=cpu does not fully support {args.dtype} well; falling back to float32.")
        dtype = torch.float32

    # 加载 tokenizer（确保与训练/数据一致）
    special_tokens = [args.eos_token] if args.eos_token else None
    tok = Tokenizer.from_files(args.vocab, args.merges, special_tokens=special_tokens)

    eos_id = None
    if args.eos_token and hasattr(tok, "special_id") and args.eos_token in tok.special_id:
        eos_id = tok.special_id[args.eos_token]

    vocab_size = len(tok.id_to_bytes)  # 用 tokenizer 推断 vocab_size，减少手填出错

    # 构建模型（超参必须和训练一致）
    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        eps=args.eps,
        device=device_t,
        dtype=dtype,
    ).to(device_t)

    model.context_lenth = args.context_length  # type: ignore[attr-defined]

    # 加载 checkpoint（train_transformer.py 存的是 {"model": state_dict, ...}）
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    # prompt -> ids
    prompt_ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device_t)  # (1, T)

    # 生成（直接用你写好的 generate）
    out_ids = generate(
        model,
        prompt_ids,
        args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_id,  # 你当前 generate 里没用它也没关系；下面我们做截断
    )

    # 如果存在 eos，就把输出截断到 eos（避免打印太长）
    out_list = out_ids[0].tolist()
    if eos_id is not None and eos_id in out_list:
        eos_pos = out_list.index(eos_id)
        out_list = out_list[: eos_pos + 1]

    print(tok.decode(out_list))


if __name__ == "__main__":
    main()
