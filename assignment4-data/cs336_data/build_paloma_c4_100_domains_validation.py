from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import tiktoken

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - 运行环境缺依赖时给清晰报错
    load_dataset = None

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - 允许只装 tiktoken
    AutoTokenizer = None


def resolve_tokenizer(tokenizer_name: str) -> tuple[str, object, int]:    # 优先用 transformers，没有就退回 gpt2 的 tiktoken
    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            return "transformers", tokenizer, tokenizer.eos_token_id
        except Exception as transformers_error:
            if tokenizer_name != "gpt2":
                raise RuntimeError(
                    f"Failed to load tokenizer {tokenizer_name!r} with transformers."
                ) from transformers_error

    if tokenizer_name == "gpt2":
        try:
            encoding = tiktoken.get_encoding("gpt2")
            return "tiktoken", encoding, encoding.eot_token
        except Exception as tiktoken_error:
            raise RuntimeError(
                "Failed to load the GPT-2 tokenizer. "
                "Try pre-downloading it first, or set HF_ENDPOINT=https://hf-mirror.com and rerun."
            ) from tiktoken_error

    raise RuntimeError(
        f"Tokenizer {tokenizer_name!r} is unavailable. "
        "Install transformers, or use --tokenizer gpt2 so the script can fall back to tiktoken."
    )


def encode_text(tokenizer_backend: str, tokenizer: object, text: str) -> list[int]:    # 把单篇文档编码成 token ids
    if tokenizer_backend == "transformers":
        return tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.encode(text)


def build_stats_path(output_path: Path) -> Path:    # 给 tokenized 输出生成统计文件路径
    return output_path.with_suffix(output_path.suffix + ".stats.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the GPT-2-tokenized Paloma C4 100 domains validation .bin file."
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to the output .bin file, e.g. /data/paloma/tokenized_paloma_c4_100_domains_validation.bin",
    )
    parser.add_argument("--dataset-name", default="allenai/paloma", help="Hugging Face dataset name")
    parser.add_argument("--subset", default="c4_100_domains", help="Subset/config name inside the dataset")
    parser.add_argument("--split", default="val", help="Split name, defaults to val")
    parser.add_argument("--field", default="text", help="Field name containing document text")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name, defaults to gpt2")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode to avoid downloading the full Arrow dataset cache first",
    )
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token")
    parser.add_argument("--cache-dir", default=None, help="Optional datasets cache directory")
    parser.add_argument("--max-documents", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--log-every", type=int, default=1000, help="Print progress every N documents")
    return parser.parse_args()


def main() -> None:    # 从 Paloma 读取 c4_100_domains validation，并编码成训练脚本兼容的 uint16 .bin
    args = parse_args()

    if load_dataset is None:
        raise RuntimeError(
            "The datasets package is required. Install it first, e.g. "
            "`pip install datasets`."
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = build_stats_path(output_path)

    tokenizer_backend, tokenizer, eos_token_id = resolve_tokenizer(args.tokenizer)
    dataset = load_dataset(
        args.dataset_name,
        args.subset,
        split=args.split,
        streaming=args.streaming,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )

    document_count = 0
    token_count = 0

    with output_path.open("wb") as fout:
        for record in dataset:
            if args.field not in record:
                raise KeyError(
                    f"Field {args.field!r} was not found in the dataset record. "
                    f"Available keys: {sorted(record.keys())}"
                )

            text = record[args.field]
            if not text or not str(text).strip():
                continue

            token_ids = encode_text(tokenizer_backend, tokenizer, str(text))
            token_ids.append(eos_token_id)

            np.array(token_ids, dtype=np.uint16).tofile(fout)

            document_count += 1
            token_count += len(token_ids)

            if document_count % args.log_every == 0:
                print(
                    f"[paloma] documents={document_count} tokens={token_count}",
                    file=sys.stderr,
                )

            if args.max_documents is not None and document_count >= args.max_documents:
                break

    stats = {
        "dataset_name": args.dataset_name,
        "subset": args.subset,
        "split": args.split,
        "field": args.field,
        "output_path": str(output_path),
        "tokenizer": args.tokenizer,
        "tokenizer_backend": tokenizer_backend,
        "eos_token_id": eos_token_id,
        "documents": document_count,
        "tokens": token_count,
        "dtype": "uint16",
        "streaming": args.streaming,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
