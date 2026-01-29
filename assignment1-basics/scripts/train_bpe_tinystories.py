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
import time
from collections import Counter
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import multiprocessing as mp

import regex as re  # 需要安装第三方 "regex" 包，才能使用 \p{L}, \p{N} 这类 Unicode 字符类别

try:
    import resource  # Linux 下：ru_maxrss 的单位是 KB
except Exception:
    resource = None  # type: ignore  # 非 Linux/不可用时：禁用资源统计

from cs336_basics.tokenizer import train_bpe_from_counts


# -------- 正则模式（GPT-2 风格）--------
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# 每个 worker 会在 initializer 里编译一次（避免重复编译）
_WORKER_PAT = None


def _init_worker():  # worker 进程初始化：把 regex 编译到全局变量，避免每个任务重复编译
    global _WORKER_PAT
    _WORKER_PAT = re.compile(PATTERN)


def _rss_kb() -> int:  # 读取当前进程 ru_maxrss（KB），用于粗略内存统计
    if resource is None:
        return 0
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# -------- 序列化辅助函数 --------
def b64e(b: bytes) -> str:  # bytes -> base64 的 ASCII 字符串（用于保存 vocab/merges）
    return base64.b64encode(b).decode("ascii")


def save_vocab_json(vocab: Dict[int, bytes], path: Path) -> None:  # 保存 vocab.json（id -> bytes 的 base64 形式）
    payload = {str(i): b64e(b) for i, b in vocab.items()}
    path.write_text(json.dumps(payload), encoding="utf-8")


def save_merges_json(merges: List[Tuple[bytes, bytes]], path: Path) -> None:  # 保存 merges.json（pair bytes 的 base64 形式）
    payload = [[b64e(a), b64e(b)] for (a, b) in merges]
    path.write_text(json.dumps(payload), encoding="utf-8")


def summarize_longest_token(vocab: Dict[int, bytes], preview: int = 200) -> str:  # 输出最长 token 信息，方便做 sanity check
    tok_id, tok_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    return (
        f"Longest token: id={tok_id}, len_bytes={len(tok_bytes)}\n"
        f"  bytes preview: {repr(tok_bytes[:preview])}\n"
        f"  utf8  preview: {tok_bytes.decode('utf-8', errors='replace')[:preview]!r}"
    )


# -------- 预分词（Pretokenization）辅助函数 --------
def _pretok_update_counts(pat: re.Pattern, docs: List[str], counts: Counter[Tuple[int, ...]]) -> None:  # 单进程：对一批 docs 做 regex 分词并更新计数
    for doc in docs:
        for m in pat.finditer(doc):
            b = m.group(0).encode("utf-8")
            counts[tuple(b)] += 1


def _worker_pretokenize_docs(docs: List[str]) -> Tuple[Counter[Tuple[int, ...]], int]:  # 多进程：worker 对一批 docs 预分词，返回计数与 rss
    global _WORKER_PAT
    assert _WORKER_PAT is not None, "Worker regex not initialized"  # worker 未初始化 regex 时直接报错

    counts: Counter[Tuple[int, ...]] = Counter()
    pat = _WORKER_PAT
    for doc in docs:
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
) -> Tuple[Counter[Tuple[int, ...]], int, int]:  # 读取文本并按 <|endoftext|> 分割，在单/多进程下做预分词统计
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    # 决定 worker 数量：0=用全部核；1=禁用多进程；>1=启用多进程
    if workers <= 0:
        workers = os.cpu_count() or 4

    total_counts: Counter[Tuple[int, ...]] = Counter()
    eot_count = 0
    max_worker_rss_kb = 0

    # 流式按分隔符切分，并用 carry 处理“分隔符跨 block 边界”的情况
    carry = ""
    batch: List[str] = []

    if workers <= 1:
        # 单进程模式（符合 --workers 1=disable multiprocessing 的语义）
        pat = re.compile(PATTERN)

        with open(in_path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(block_chars)
                if not chunk:
                    break
                data = carry + chunk
                parts = data.split(eot_token)
                carry = parts.pop()  # 最后一段可能是被截断的 doc

                # parts 中每个 split 边界都对应一个 EOT
                eot_count += len(parts)

                for doc in parts:
                    batch.append(doc)
                    if len(batch) >= docs_per_task:
                        _pretok_update_counts(pat, batch, total_counts)
                        batch = []

        # 处理最后剩余的一段 doc
        batch.append(carry)
        _pretok_update_counts(pat, batch, total_counts)

        # 单进程模式没有 worker rss
        max_worker_rss_kb = 0

    else:
        # 多进程模式（优先 fork；不支持则使用 spawn）
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")

        pool = ctx.Pool(processes=workers, initializer=_init_worker, maxtasksperchild=200)

        pending = deque()  # 使用 deque 避免 list.pop(0) 的 O(n) 开销
        try:
            def flush_one():
                nonlocal max_worker_rss_kb, total_counts
                res = pending.popleft().get()
                c, rss_kb = res
                total_counts.update(c)
                if rss_kb > max_worker_rss_kb:
                    max_worker_rss_kb = rss_kb

            with open(in_path, "r", encoding="utf-8") as f:
                while True:
                    chunk = f.read(block_chars)
                    if not chunk:
                        break
                    data = carry + chunk
                    parts = data.split(eot_token)
                    carry = parts.pop()

                    eot_count += len(parts)

                    for doc in parts:
                        batch.append(doc)
                        if len(batch) >= docs_per_task:
                            pending.append(pool.apply_async(_worker_pretokenize_docs, (batch,)))
                            batch = []

                            # 节流：避免 pending 太多导致内存膨胀
                            while len(pending) >= max_pending_tasks:
                                flush_one()

            # 处理最后剩余的一段 doc
            batch.append(carry)
            pending.append(pool.apply_async(_worker_pretokenize_docs, (batch,)))
            batch = []

            # 收集并合并剩余任务的结果
            while pending:
                flush_one()

        finally:
            pool.close()
            pool.join()

    # 把 special token 的出现次数作为“原子 token”计数（不会被 pretokenize 成 bytes 序列）
    if eot_count > 0:
        total_counts[(special_id,)] += eot_count

    return total_counts, eot_count, max_worker_rss_kb


def main() -> int:  # 训练入口：预分词统计 -> BPE 训练 -> 保存 vocab/merges/meta 并打印摘要
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TinyStories 训练集 txt 路径")
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--out", default="artifacts/tinystories_bpe_mp")
    ap.add_argument("--special", default="<|endoftext|>", help="EOT special token 字符串")
    ap.add_argument("--workers", type=int, default=0, help="0=使用全部 CPU 核；1=禁用多进程")
    ap.add_argument("--docs-per-task", type=int, default=256, help="每个 worker 任务处理的 doc 数（批大小）")
    ap.add_argument("--block-chars", type=int, default=8 * 1024 * 1024, help="每次读取的字符块大小")
    ap.add_argument("--max-pending", type=int, default=64, help="限制排队任务数量以控制内存")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    eot = args.special
    special_tokens = [eot]
    special_id = 256  # 约定第一个 special token 从 256 开始

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

    # 保存输出产物
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
            "pretokenize": t1 - t0,
            "bpe_train": t2 - t1,
            "total": t2 - t0,
        },
        "rss_kb": {
            "main_process_max": int(main_rss_kb),
            "max_worker_ru_maxrss": int(max_worker_rss_kb),
            "note": "要做完整内存统计（包含子进程），建议在 Python 外用 /usr/bin/time -v。",
        },
        "outputs": {
            "vocab_json": str(vocab_path),
            "merges_json": str(merges_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 打印摘要信息
    print("=== TinyStories BPE（预分词 + 训练）===")
    print(f"input: {in_path}")
    print(f"vocab_size: {args.vocab_size}")
    print(f"special: {eot} (id={special_id}), count_in_file={eot_count}")
    if args.workers <= 1:
        print("mode: 单进程预分词")
        print("workers: 1")
    else:
        print("mode: 多进程预分词")
        print(f"workers: {args.workers if args.workers > 0 else (os.cpu_count() or 4)}")
    print(f"time pretokenize : {t1 - t0:.2f}s")
    print(f"time bpe_train   : {t2 - t1:.2f}s")
    print(f"time total       : {t2 - t0:.2f}s")

    if resource is not None:
        print(f"main ru_maxrss: {main_rss_kb / 1024 / 1024:.2f} GB")
        if args.workers > 1:
            print(f"max worker ru_maxrss (approx): {max_worker_rss_kb / 1024 / 1024:.2f} GB")

    print(summarize_longest_token(vocab))
    print(f"saved: {vocab_path}")
    print(f"saved: {merges_path}")
    print(f"saved: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
