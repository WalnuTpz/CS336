from __future__ import annotations

import os
import regex as re
from collections import Counter, defaultdict

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    special_id : dict[str, int]
) -> Counter[tuple[int, ...]]:
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
    pretoken_counts: Counter[tuple[int, ...]] = Counter()
    for part in parts:
        if not part:
            continue
        if part in special_set:
            pretoken_counts[(special_id[part], )] += 1 # special token 转换为它在词表中的 id 后累加出现次数
        else:
            for m in PAT.finditer(part):
                b = m.group(0).encode("utf-8")
                pretoken_counts[tuple(b)] += 1
                # 普通段内按照 PAT 分割成若干个 token，每个 token 转换为 bytes 后再转换为 tuple，累加出现次数

    return pretoken_counts

def merge(
    pretoken_counts: Counter[tuple[int, ...]],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> tuple[Counter[tuple[int, ...]], dict[int, bytes], list[tuple[bytes, bytes]]]:
    pair_counts = defaultdict(int)
    for seq, freq in pretoken_counts.items():   # 某个 pretoken 对应的整数序列和它出现的次数
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            pair_counts[(seq[i], seq[i + 1])] += freq     # 这个 pair 对总次数的贡献是 freq

    if not pair_counts:
        return pretoken_counts, vocab, merges
    best_pair, _ = max(
        pair_counts.items(),
        key=lambda kv: (kv[1], vocab[kv[0][0]], vocab[kv[0][1]]))
    # 找出出现次数最多的 pair （比较 kv[1]），若次数相同则选择 pair 最大的那个（优先比较第一个元素，在比较第二个元素）

    a, b = best_pair
    new_tok = vocab[a] + vocab[b]     # 新的 token
    new_id = len(vocab)
    vocab[new_id] = new_tok
    merges.append((vocab[a], vocab[b]))

    new_pretoken_counts : Counter[tuple[int, ...]] = Counter()
    for seq, freq in pretoken_counts.items():
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                new_seq.append(new_id)      # 可以合并，则添加合并后的 new_id，并且 i 要额外加一
                i += 2
            else:
                new_seq.append(seq[i])      # 否则添加原来的 id
                i += 1
        new_pretoken_counts[tuple(new_seq)] += freq     # 累加新的 seq 出现的次数

    return new_pretoken_counts, vocab, merges

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
    special_id : dict[str, int] = {}

    for i in range(256):
        vocab[i] = bytes([i])   # 初始字符加入词表
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")  # 特殊字符加入词表
        special_id[special_tokens[i]] = i + 256

    pretoken_counts = pretokenize_file(input_path, special_tokens, special_id)

    num_merges = vocab_size - 256 - len(special_tokens)
    for i in range(num_merges):
        pretoken_counts, vocab, merges = merge(pretoken_counts, vocab, merges)

    return vocab, merges


