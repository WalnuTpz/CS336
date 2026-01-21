from __future__ import annotations

import os
import regex as re
from collections import Counter, defaultdict

def pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str]
) -> tuple[list[tuple[bytes, ...]], set[bytes]]:
    with open(input_path, "r", encoding="utf-8") as f:  # 读入文本
        text = f.read()

    # 将原文本按照 special_token 进行分割
    if special_tokens:
        delimit = "|".join(re.escape(tok) for tok in special_tokens)
        # 把每个 special_token 的特殊字符加反斜杠转义以后再用 '|' 连接起来
        parts = re.split(f"({delimit})", text)  # 用分隔符切割原文本，并让分隔符也出现在结果里
        special_set = set(special_tokens)
        special_byte_set = {tok.encode("utf-8") for tok in special_tokens}
    else:
        parts = [text]
        special_set = set()
        special_byte_set = set()

    # 将原文本进一步进行 pre-tokenization
    pre_tokens: list[tuple[bytes, ...]] = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for part in parts:
        if not part:
            continue
        if part in special_set:
            b = part.encode("utf-8")
            pre_tokens.append((b,))  # special token 变为bytes后作为单元素 tuple 加入
        else:
            for m in re.finditer(PAT, part):
                b = m.group(0).encode("utf-8")
                pre_tokens.append(tuple(b[i:i+1] for i in range(len(b))))
                # 普通段内按照 PAT 查找所有 token，每个 token 内部拆分为单元素 bytes，然后转换为 tuple 加入

    return pre_tokens, special_byte_set

def merge(
    pre_tokens: list[tuple[bytes, ...]],
    special_byte_set : set[bytes],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> tuple[list[tuple[bytes, ...]], dict[int, bytes], list[tuple[bytes, bytes]]]:
    counts = defaultdict(int)
    for pre_token in pre_tokens:
        if len(pre_token) == 1 or pre_token[0] in special_byte_set:
            continue
        for a, b in zip(pre_token, pre_token[1 :]):
            pair = (a, b)
            counts[pair] += 1

    best_count = 0
    best_pair = None
    for pair, count in counts.items():
        if count > best_count:
            best_pair = pair
            best_count = count
        elif count == best_count and pair > best_pair:
            best_pair = pair

    if not counts:
        return pre_tokens, vocab, merges
    a, b = best_pair
    new_tok = a + b
    new_id = len(vocab)
    vocab[new_id] = new_tok
    merges.append((a, b))

    new_pre_tokens : list[tuple[bytes, ...]] = []
    for pre_token in pre_tokens:
        if len(pre_token) == 1:
            new_pre_tokens.append(pre_token)    # 长度为 1 的直接加入 tokens
            continue
        i = 0
        new_pre_token = []
        while i < len(pre_token):
            if i < len(pre_token) - 1 and pre_token[i] == a and pre_token[i + 1] == b:
                new_pre_token.append(new_tok)   # 将 (a, b) 合并后加入 token
                i += 2
            else:
                new_pre_token.append(pre_token[i])    # 剩下的也直接加入 token
                i += 1
        new_pre_tokens.append(tuple(new_pre_token))    # token 转换为 tuple 类型以后再加入 tokens

    return new_pre_tokens, vocab, merges

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

    pre_tokens, special_set = pretokenize_file(input_path, special_tokens)

    num_merges = vocab_size - 256 - len(special_tokens)
    for i in range(num_merges):
        pre_tokens, vocab, merges = merge(pre_tokens, special_set, vocab, merges)

    return vocab, merges


