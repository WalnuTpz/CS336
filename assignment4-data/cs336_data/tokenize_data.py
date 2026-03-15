from __future__ import annotations

import argparse
import glob
import json
import multiprocessing
import os
from pathlib import Path
import sys

import numpy as np
import tiktoken
from transformers import AutoTokenizer
from xopen import xopen


TOKENIZER = None
EOS_TOKEN_ID = None
TOKENIZER_BACKEND = None


def expand_input_paths(inputs: list[str]) -> list[str]:    # 展开 glob，并去重排序
    input_paths: set[str] = set()

    for item in inputs:
        matches = glob.glob(item)
        if matches:
            input_paths.update(matches)
        else:
            input_paths.add(item)

    return sorted(input_paths)


def resolve_tokenizer_backend(tokenizer_name: str) -> tuple[str, int]:    # 先在主进程准备 tokenizer，避免每个 worker 都去联网
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return "transformers", tokenizer.eos_token_id
    except Exception as transformers_error:
        if tokenizer_name == "gpt2":
            try:
                encoding = tiktoken.get_encoding("gpt2")
                return "tiktoken", encoding.eot_token
            except Exception as tiktoken_error:
                raise RuntimeError(
                    "Failed to load the GPT-2 tokenizer. "
                    "Try pre-downloading it first, or set HF_ENDPOINT=https://hf-mirror.com and rerun."
                ) from tiktoken_error
        raise RuntimeError(
            f"Failed to load tokenizer {tokenizer_name!r}. "
            "If this is a HuggingFace model, try pre-downloading it first."
        ) from transformers_error


def init_tokenizer(tokenizer_name: str, tokenizer_backend: str) -> None:    # 在每个 worker 里各自加载一次 tokenizer
    global TOKENIZER
    global EOS_TOKEN_ID
    global TOKENIZER_BACKEND

    TOKENIZER_BACKEND = tokenizer_backend

    if tokenizer_backend == "transformers":
        TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        EOS_TOKEN_ID = TOKENIZER.eos_token_id
        return

    TOKENIZER = tiktoken.get_encoding("gpt2")
    EOS_TOKEN_ID = TOKENIZER.eot_token


def tokenize_line_and_add_eos(line: str) -> list[int]:    # 对一篇文档做 tokenization，并在末尾补 eos
    text = line.rstrip("\n")
    if not text:
        return []

    token_ids = TOKENIZER.encode(text)
    token_ids.append(EOS_TOKEN_ID)
    return token_ids


def iter_documents(input_paths: list[str]):    # 顺序读取多份过滤后文本，一行就是一篇文档
    for input_path in input_paths:
        with xopen(input_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line


def build_stats_path(output_path: Path) -> Path:    # 给 tokenized 输出生成统计文件路径
    return output_path.with_suffix(output_path.suffix + ".stats.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize filtered LM data with the GPT-2 tokenizer.")
    parser.add_argument("inputs", nargs="+", help="Filtered text files or glob patterns")
    parser.add_argument("--output-path", required=True, help="Path to the output .bin file")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name, defaults to gpt2")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="Number of tokenizer workers")
    parser.add_argument("--chunksize", type=int, default=100, help="chunksize for multiprocessing imap")
    parser.add_argument("--log-every", type=int, default=10000, help="Print progress every N documents")
    return parser.parse_args()


def main() -> None:    # 命令行入口：把过滤后文本编码成 uint16，并写成训练脚本兼容的 .bin
    args = parse_args()
    input_paths = expand_input_paths(args.inputs)
    if not input_paths:
        raise ValueError("No filtered input files matched the provided patterns.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = build_stats_path(output_path)

    document_count = 0
    token_count = 0
    tokenizer_backend, eos_token_id = resolve_tokenizer_backend(args.tokenizer)
    if tokenizer_backend == "tiktoken":
        print(
            "[tokenize] transformers tokenizer files are unavailable; falling back to tiktoken gpt2 encoding",
            file=sys.stderr,
        )

    with multiprocessing.Pool(
        processes=max(1, args.workers),
        initializer=init_tokenizer,
        initargs=(args.tokenizer, tokenizer_backend),
    ) as pool, output_path.open("wb") as fout:
        for token_ids in pool.imap(tokenize_line_and_add_eos, iter_documents(input_paths), chunksize=args.chunksize):
            if not token_ids:
                continue

            ids_array = np.array(token_ids, dtype=np.uint16)
            ids_array.tofile(fout)

            document_count += 1
            token_count += len(token_ids)

            if document_count % args.log_every == 0:
                print(
                    f"[tokenize] documents={document_count} tokens={token_count}",
                    file=sys.stderr,
                )

    stats = {
        "input_files": input_paths,
        "output_path": str(output_path),
        "tokenizer": args.tokenizer,
        "tokenizer_backend": tokenizer_backend,
        "eos_token_id": eos_token_id,
        "documents": document_count,
        "tokens": token_count,
        "dtype": "uint16",
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
