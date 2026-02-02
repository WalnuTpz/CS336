from __future__ import annotations

import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.generation import generate


# -------- 你只需要改这里：路径 + 超参（必须和训练一致） --------
CKPT_PATH = ""
VOCAB_PATH = ""
MERGES_PATH = ""

CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 1344
ROPE_THETA = 10000.0
EPS = 1e-5

DTYPE = torch.float32  # 想用 bfloat16/float16 也行（CPU 上建议 float32）


def main() -> None:
    # -------- device --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- tokenizer（要和训练/数据一致）--------
    tok = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])
    eos_id = tok.special_id.get("<|endoftext|>", None)

    vocab_size = len(tok.id_to_bytes)

    # -------- model（超参要和训练一致）--------
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
        eps=EPS,
        device=device,
        dtype=DTYPE,
    ).to(device)

    # -------- load checkpoint --------
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    print("\n[ready] 输入 prompt 直接生成；输入 /q 退出；输入 /h 看帮助\n")

    # -------- REPL --------
    temperature = 1.2
    top_p = 0.9
    max_new_tokens = 128

    while True:
        s = input("prompt> ").rstrip("\n")
        if not s:
            continue
        if s in ("/q", "/quit", "quit", "exit"):
            break
        if s in ("/h", "/help"):
            print(
                "命令：\n"
                "  /q 退出\n"
                "  /set temp=0.8 top_p=0.9 max_new=128  （修改采样参数）\n"
            )
            continue
        if s.startswith("/set "):
            # 极简解析：/set temp=... top_p=... max_new=...
            parts = s[len("/set ") :].split()
            for kv in parts:
                k, v = kv.split("=")
                if k == "temp":
                    temperature = float(v)
                elif k == "top_p":
                    top_p = float(v)
                elif k in ("max_new", "max_new_tokens"):
                    max_new_tokens = int(v)
            print(f"[set] temp={temperature} top_p={top_p} max_new_tokens={max_new_tokens}")
            continue

        prompt_ids = torch.tensor([tok.encode(s)], dtype=torch.long, device=device)  # (1, T)

        out_ids = generate(
            model,
            prompt_ids,
            max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_id,
        )

        # 截断到 eos（防止打印很长）
        ids = out_ids[0].tolist()
        if eos_id is not None and eos_id in ids:
            ids = ids[: ids.index(eos_id) + 1]

        print(tok.decode(ids))
        print("-" * 60)


if __name__ == "__main__":
    main()
