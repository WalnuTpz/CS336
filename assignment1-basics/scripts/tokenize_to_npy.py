from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:  # 解析命令行参数并返回配置
    p = argparse.ArgumentParser()
    p.add_argument("--input_txt", type=str, required=True)
    p.add_argument("--output_npy", type=str, required=True)
    p.add_argument("--vocab_json", type=str, required=True)
    p.add_argument("--merges_json", type=str, required=True)

    # 可选：加入文档结束符（例如 <|endoftext|>）
    p.add_argument("--add_eot", action="store_true")
    p.add_argument("--eot_token", type=str, default="<|endoftext|>")

    # 输出 dtype（token id 一般 int32 足够）
    p.add_argument("--dtype", type=str, default="int32", choices=["int32", "int64"])

    return p.parse_args()


def count_tokens(input_txt: str, tok: Tokenizer, add_eot: bool) -> int:  # 第一遍扫描：统计总 token 数（用于预分配输出数组）
    total = 0
    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            # 去掉末尾换行，避免把 '\n' 也当成内容编码进去
            line = line.rstrip("\n")

            # 空行允许存在：此时 ids_len=0；是否额外加 EOT 由 add_eot 决定
            ids_len = len(tok.encode(line)) if line else 0

            total += ids_len
            if add_eot:
                total += 1
    return total


def tokenize_to_npy(
    input_txt: str,
    output_npy: str,
    tok: Tokenizer,
    add_eot: bool,
    eot_id: Optional[int],
    out_dtype: np.dtype,
) -> None:  # 两遍扫描：将文本编码为 token ids，并以 memmap 方式写入 .npy（内存友好）
    # 第一遍：只统计长度（为了 open_memmap 需要 shape）
    n_tokens = count_tokens(input_txt, tok, add_eot)
    if n_tokens <= 0:
        raise ValueError("No tokens produced; check your input file or tokenizer.")

    # 创建可 mmap 的 .npy 文件并写入
    arr = np.lib.format.open_memmap(output_npy, mode="w+", dtype=out_dtype, shape=(n_tokens,))

    pos = 0
    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            ids = tok.encode(line) if line else []

            if ids:
                arr[pos : pos + len(ids)] = np.asarray(ids, dtype=out_dtype)
                pos += len(ids)

            if add_eot:
                if eot_id is None:
                    raise ValueError("add_eot=True but eot_id is None.")
                arr[pos] = eot_id
                pos += 1

    if pos != n_tokens:
        raise RuntimeError(f"Token count mismatch: wrote {pos}, expected {n_tokens}.")

    # 确保落盘
    arr.flush()


def main() -> None:  # 脚本入口：加载 tokenizer，执行 tokenize，并输出 vocab_size 等信息
    args = parse_args()
    out_dtype = np.int32 if args.dtype == "int32" else np.int64

    # 构造 tokenizer：from_files 会读取 vocab/merges（json 内部 base64 编码）
    special_tokens = [args.eot_token] if args.add_eot else None
    tok = Tokenizer.from_files(args.vocab_json, args.merges_json, special_tokens=special_tokens)

    eot_id = None
    if args.add_eot:
        # Tokenizer 会把 special token 映射到 special_id 里
        eot_id = tok.special_id[args.eot_token]

    os.makedirs(os.path.dirname(args.output_npy) or ".", exist_ok=True)

    tokenize_to_npy(
        input_txt=args.input_txt,
        output_npy=args.output_npy,
        tok=tok,
        add_eot=args.add_eot,
        eot_id=eot_id,
        out_dtype=out_dtype,
    )

    # vocab_size 就是 id_to_bytes 的大小
    print(f"[done] wrote: {args.output_npy}")
    print(f"[info] vocab_size = {len(tok.id_to_bytes)}")


if __name__ == "__main__":
    main()
