#!/usr/bin/env python3
"""
Train byte-level BPE on TinyStories (or any text file), then serialize vocab + merges.

Usage (from repo root):
  uv run python scripts/train_bpe_tinystories.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --vocab-size 10000 \
    --out artifacts/tinystories_bpe \
    --special "<|endoftext|>"

Optional profiling:
  uv run python scripts/train_bpe_tinystories.py ... --profile
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    # Preferred (repo-style)
    from cs336_basics.tokenizer import train_bpe  # type: ignore
except Exception:
    # Fallback (if you're running next to tokenizer.py directly)
    from tokenizer import train_bpe  # type: ignore

# ru_maxrss (Linux): kilobytes
try:
    import resource
except Exception:
    resource = None  # type: ignore


BytesVocab = Dict[int, bytes]
Merges = List[Tuple[bytes, bytes]]


def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def save_vocab_json(vocab: BytesVocab, path: Path) -> None:
    # Store bytes as base64 strings: { "id": "base64..." }
    payload = {str(i): b64e(b) for i, b in vocab.items()}
    path.write_text(json.dumps(payload), encoding="utf-8")


def save_merges_json(merges: Merges, path: Path) -> None:
    # Store as list of [base64(a), base64(b)] in merge order
    payload = [[b64e(a), b64e(b)] for (a, b) in merges]
    path.write_text(json.dumps(payload), encoding="utf-8")


def summarize_longest_token(vocab: BytesVocab, preview: int = 200) -> str:
    tok_id, tok_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    utf8_preview = tok_bytes.decode("utf-8", errors="replace")[:preview]
    bytes_preview = repr(tok_bytes[:preview])
    return (
        f"Longest token id={tok_id}, len_bytes={len(tok_bytes)}\n"
        f"  bytes preview: {bytes_preview}\n"
        f"  utf8  preview: {utf8_preview!r}"
    )


def max_rss_gb() -> float | None:
    if resource is None:
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # Linux ru_maxrss is KB
    return float(ru.ru_maxrss) / 1024.0 / 1024.0


def run_train(args: argparse.Namespace) -> int:
    in_path = Path(args.input).expanduser()
    if not in_path.exists():
        print(f"[ERROR] input file not found: {in_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = args.special or []
    vocab_size = int(args.vocab_size)

    # ---- Train ----
    t0 = time.time()
    vocab, merges = train_bpe(str(in_path), vocab_size, special_tokens)
    t1 = time.time()

    elapsed_s = t1 - t0
    elapsed_h = elapsed_s / 3600.0

    # ---- Save artifacts ----
    vocab_path = out_dir / "vocab.json"
    merges_path = out_dir / "merges.json"
    meta_path = out_dir / "meta.json"

    save_vocab_json(vocab, vocab_path)
    save_merges_json(merges, merges_path)

    meta = {
        "input": str(in_path),
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "elapsed_seconds": elapsed_s,
        "elapsed_hours": elapsed_h,
        "max_rss_gb": max_rss_gb(),
        "vocab_json": str(vocab_path),
        "merges_json": str(merges_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- Print summary ----
    print("=== BPE training done ===")
    print(f"input: {in_path}")
    print(f"vocab_size: {vocab_size}")
    print(f"special_tokens: {special_tokens}")
    print(f"elapsed: {elapsed_s:.2f}s ({elapsed_h:.4f}h)")
    rss = max_rss_gb()
    if rss is not None:
        print(f"max_rss: {rss:.2f} GB (process peak)")
    print(summarize_longest_token(vocab))
    print(f"saved: {vocab_path}")
    print(f"saved: {merges_path}")
    print(f"saved: {meta_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="Path to TinyStories train text (e.g., data/TinyStoriesV2-GPT4-train.txt)",
    )
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--out", default="artifacts/tinystories_bpe")
    ap.add_argument(
        "--special",
        action="append",
        default=[],
        help='Special token to include (repeatable). Default usage: --special "<|endoftext|>"',
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="Run under cProfile and write out/profile.pstats",
    )
    args = ap.parse_args()

    if args.profile:
        import cProfile
        import pstats

        out_dir = Path(args.out).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        prof_path = out_dir / "profile.pstats"
        cProfile.runctx("run_train(args)", globals(), locals(), str(prof_path))
        print(f"[profile] wrote: {prof_path}")
        # Print top 30 by cumulative time
        p = pstats.Stats(str(prof_path))
        p.sort_stats("cumtime").print_stats(30)
        return 0

    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""
训练示例：
uv run python scripts/train_bpe_tinystories.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --out artifacts/tinystories_bpe \
  --special "<|endoftext|>"
"""