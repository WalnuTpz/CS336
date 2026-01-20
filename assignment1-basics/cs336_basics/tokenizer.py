from __future__ import annotations

import os
import regex as re
from typing import Tuple
from collections import Counter


def pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str]
) -> list[str]:
    with open(input_path, "r", encoding="utf-8") as f:  # 读入文本
        text = f.read()

    # 将原文本按照 special_token 进行分割
    if special_tokens:
        delimit = "|".join(re.escape(tok) for tok in special_tokens)
        # 把每个 special_token 的特殊字符加反斜杠转义以后再用 '|' 连接起来
        parts = re.split(f"({delimit})", text)  # 用分隔符切割原文本，并让分隔符也出现在结果里
        special_set = set(special_tokens)
    else:
        parts = [text]
        special_set = set()

    # 将原文本进一步进行 pre-tokenization
    pre_tokens: list[str] = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for part in parts:
        if not part:
            continue
        if part in special_set:
            pre_tokens.append(part)  # special token 作为整体加入
        else:
            for m in re.finditer(PAT, part):
                pre_tokens.append(m.group(0))  # 普通段内按照 PAT 查找所有 token，并逐个加入

    return pre_tokens

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

    pre_tokens = pretokenize_file(input_path, special_tokens)

    num_merges = vocab_size - 256 - len(special_tokens)
    for i in range(num_merges):
        merge()
    return (vocab, merges)
