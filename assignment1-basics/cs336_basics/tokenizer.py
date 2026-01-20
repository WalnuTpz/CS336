from __future__ import annotations

import os
import re
import regex
from typing import Tuple

def merge():
    return

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Returns:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
    """
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    for i in range(256):
        vocab[i] = bytes([i])   # 初始字符加入词表
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")  # 特殊字符加入词表

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    if special_tokens:
        delimit = '|'.join(re.escape(tok) for tok in special_tokens)
        # 把每个 special_token 的特殊字符加反斜杠转义以后再用 '|' 连接起来
        parts = re.split(f"({delimit})", text)  # 用分隔符切割原文本，并让分隔符也出现在结果里
    else:
        parts = [text]

    pretokens : list[str] = []
    special_set = set(special_tokens)  # 用 set 加速 in 判断
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for part in parts:
        if not part:
            continue
        if part in special_set:
            pretokens.append(part)  # special token 作为整体加入
        else:
            for m in regex.finditer(PAT, part):
                pretokens.append(m.group(0))    # 在 part 内按照 PAT 查找所有 token，并逐个加入





    num_merges = vocab_size - 256 - len(special_tokens)
    for i in range(num_merges):
        merge()
    return (vocab, merges)
