#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train byte-level BPE on TinyStories with multiprocessing pretokenization.

Key tricks (per assignment hint):
1) Treat <|endoftext|> as document separator in the raw file: split on it.
2) Treat <|endoftext|> as a SPECIAL CASE before applying BPE merges:
   - It is NOT pretokenized into bytes.
   - It is counted as an atomic token id: (special_id,)

This avoids cross-document merges and speeds up pretokenization by parallelism.

Usage (repo root):
  /usr/bin/time -v uv run python scripts/train_bpe_tinystories.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --vocab-size 10000 \
    --out artifacts/tinystories_bpe_mp \
    --special "<|endoftext|>" \
    --workers 0

Profiling note:
- With multiprocessing, cProfile in main won't include worker time.
  This script prints phase timings (pretokenize vs merge-train).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from cs336_basics.tokenizer import train_bpe_from_counts


import regex as re  # needs "regex" package for \p{L}, \p{N}

try:
    import resource  # Linux: ru_maxrss in KB
except Exception:
    resource = None  # type: ignore

import multiprocessing as mp
import heapq


# -------- Regex pattern (GPT-2 style) --------
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# Compiled per-worker in initializer
_WORKER_PAT = None


def _init_worker():
    global _WORKER_PAT
    _WORKER_PAT = re.compile(PATTERN)


def _rss_kb() -> int:
    if resource is None:
        return 0
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# -------- Serialization helpers --------
def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def save_vocab_json(vocab: Dict[int, bytes], path: Path) -> None:
    payload = {str(i): b64e(b) for i, b in vocab.items()}
    path.write_text(json.dumps(payload), encoding="utf-8")


def save_merges_json(merges: List[Tuple[bytes, bytes]], path: Path) -> None:
    payload = [[b64e(a), b64e(b)] for (a, b) in merges]
    path.write_text(json.dumps(payload), encoding="utf-8")


def summarize_longest_token(vocab: Dict[int, bytes], preview: int = 200) -> str:
    tok_id, tok_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    return (
        f"Longest token: id={tok_id}, len_bytes={len(tok_bytes)}\n"
        f"  bytes preview: {repr(tok_bytes[:preview])}\n"
        f"  utf8  preview: {tok_bytes.decode('utf-8', errors='replace')[:preview]!r}"
    )


# -------- Multiprocessing pretokenization --------
def _worker_pretokenize_docs(docs: List[str]) -> Tuple[Counter[Tuple[int, ...]], int]:
    """
    Pretokenize docs (each doc is a standalone string WITHOUT <|endoftext|>).
    Returns:
      - Counter[tuple[int,...]]: seq of byte-ids 0..255 -> count
      - worker ru_maxrss (KB)
    """
    global _WORKER_PAT
    assert _WORKER_PAT is not None, "Worker regex not initialized"

    counts: Counter[Tuple[int, ...]] = Counter()
    pat = _WORKER_PAT
    for doc in docs:
        # doc is independent: no cross-doc tokenization
        for m in pat.finditer(doc):
            b = m.group(0).encode("utf-8")
            counts[tuple(b)] += 1
    return counts, _rss_kb()


def pretokenize_file_mp(
    input_path: str,
    eot_token: str,
    special_id: int,
    workers: int,
    docs_per_task: int,
    block_chars: int,
    max_pending_tasks: int,
) -> Tuple[Counter[Tuple[int, ...]], int, int]:
    """
    Stream-read file, split by eot_token, send docs in batches to worker pool.
    We count eot occurrences in the main process and finally add:
      counts[(special_id,)] += eot_count

    Returns:
      - word_counts: Counter[tuple[int,...]] for normal bytes (and special_id singleton added)
      - eot_count: number of <|endoftext|> in file
      - max_worker_rss_kb: maximum ru_maxrss among workers (KB)
    """
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    # Decide worker count
    if workers <= 0:
        workers = os.cpu_count() or 4

    # Prefer fork on Linux for speed; fall back otherwise
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    pool = ctx.Pool(processes=workers, initializer=_init_worker, maxtasksperchild=200)

    pending: List[mp.pool.ApplyResult] = []
    total_counts: Counter[Tuple[int, ...]] = Counter()
    eot_count = 0
    max_worker_rss_kb = 0

    def flush_one():
        nonlocal max_worker_rss_kb, total_counts
        res = pending.pop(0).get()
        c, rss_kb = res
        total_counts.update(c)
        if rss_kb > max_worker_rss_kb:
            max_worker_rss_kb = rss_kb

    # Streaming split-by-delimiter with carry to handle delimiter across block boundaries
    carry = ""
    batch: List[str] = []

    with open(in_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(block_chars)
            if not chunk:
                break
            data = carry + chunk
            parts = data.split(eot_token)
            carry = parts.pop()  # last piece may be partial doc

            # Every split boundary here corresponds to one EOT in the input
            eot_count += len(parts)

            for doc in parts:
                batch.append(doc)
                if len(batch) >= docs_per_task:
                    pending.append(pool.apply_async(_worker_pretokenize_docs, (batch,)))
                    batch = []

                    # Throttle to avoid too much queued memory
                    while len(pending) >= max_pending_tasks:
                        flush_one()

        # Final doc (after last delimiter). Even if empty, it's harmless.
        batch.append(carry)
        pending.append(pool.apply_async(_worker_pretokenize_docs, (batch,)))
        batch = []

    # Collect remaining
    while pending:
        flush_one()

    pool.close()
    pool.join()

    # Add special token occurrences as atomic singleton token id
    if eot_count > 0:
        total_counts[(special_id,)] += eot_count

    return total_counts, eot_count, max_worker_rss_kb

# -------- Main script --------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TinyStories train txt path")
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--out", default="artifacts/tinystories_bpe_mp")
    ap.add_argument("--special", default="<|endoftext|>", help="EOT special token string")
    ap.add_argument("--workers", type=int, default=0, help="0=use all cores; 1=disable multiprocessing")
    ap.add_argument("--docs-per-task", type=int, default=256, help="Batch docs per worker task")
    ap.add_argument("--block-chars", type=int, default=8 * 1024 * 1024, help="Read block size in characters")
    ap.add_argument("--max-pending", type=int, default=64, help="Throttle queued tasks to limit memory")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    eot = args.special
    special_tokens = [eot]
    special_id = 256  # by convention: first special token at 256

    t0 = time.time()
    wc, eot_count, max_worker_rss_kb = pretokenize_file_mp(
        input_path=str(in_path),
        eot_token=eot,
        special_id=special_id,
        workers=args.workers,
        docs_per_task=args.docs_per_task,
        block_chars=args.block_chars,
        max_pending_tasks=args.max_pending,
    )
    t1 = time.time()

    vocab, merges = train_bpe_from_counts(
        pretoken_counts=wc,
        vocab_size=int(args.vocab_size),
        special_tokens=special_tokens,
    )

    t2 = time.time()

    # Save artifacts
    vocab_path = out_dir / "vocab.json"
    merges_path = out_dir / "merges.json"
    meta_path = out_dir / "meta.json"

    save_vocab_json(vocab, vocab_path)
    save_merges_json(merges, merges_path)

    main_rss_kb = _rss_kb()
    meta = {
        "input": str(in_path),
        "vocab_size": int(args.vocab_size),
        "special_tokens": special_tokens,
        "special_id_map": {eot: special_id},
        "eot_count_in_file": int(eot_count),
        "timing_seconds": {
            "pretokenize_mp": t1 - t0,
            "bpe_train": t2 - t1,
            "total": t2 - t0,
        },
        "rss_kb": {
            "main_process_max": int(main_rss_kb),
            "max_worker_ru_maxrss": int(max_worker_rss_kb),
            "note": "For full accounting (incl. children), prefer /usr/bin/time -v outside Python.",
        },
        "outputs": {
            "vocab_json": str(vocab_path),
            "merges_json": str(merges_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Print summary
    print("=== TinyStories BPE (mp pretokenize) ===")
    print(f"input: {in_path}")
    print(f"vocab_size: {args.vocab_size}")
    print(f"special: {eot} (id={special_id}), count_in_file={eot_count}")
    print(f"workers: {args.workers if args.workers > 0 else (os.cpu_count() or 4)}")
    print(f"time pretokenize_mp: {t1 - t0:.2f}s")
    print(f"time bpe_train     : {t2 - t1:.2f}s")
    print(f"time total         : {t2 - t0:.2f}s")
    if resource is not None:
        print(f"main ru_maxrss: {main_rss_kb / 1024 / 1024:.2f} GB")
        print(f"max worker ru_maxrss (approx): {max_worker_rss_kb / 1024 / 1024:.2f} GB")
    print(summarize_longest_token(vocab))
    print(f"saved: {vocab_path}")
    print(f"saved: {merges_path}")
    print(f"saved: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

