from __future__ import annotations

import os
import regex as re
import heapq
from collections import Counter, defaultdict

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    special_id: dict[str, int]
) -> Counter[tuple[int, ...]]:
    with open(input_path, "r", encoding="utf-8") as f:  # 读入文本
        text = f.read()

    # 将原文本按照 special_token 进行分割
    if special_tokens and any(tok in text for tok in special_tokens):
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
                # 普通段内按照 PAT 分割成若干个 token，每个 token 解码为 bytes 后再转换为 tuple，累加出现次数
                # 因为直接取出单个 bytes 得到的是它的 int 值，所以此时的 tuple 里面也都是 int，它也就是后面的 seq
                b = m.group(0).encode("utf-8")
                pretoken_counts[tuple(b)] += 1

    return pretoken_counts

class RevBytes:     # 反转字节类（用于后续建立大根堆）
    __slots__ = ("b",)
    def __init__(self, b: bytes): self.b = b
    def __lt__(self, other: "RevBytes") -> bool:
        return self.b > other.b   # 反转：让更大的 bytes “更小”
    def __eq__(self, other: object) -> bool:
        return isinstance(other, RevBytes) and self.b == other.b

def merge(
    pretoken_counts: Counter[tuple[int, ...]],
    pair_counts: dict[tuple[int,int], int],
    pair_sets: dict[tuple[int,int], set[tuple[int,...]]],
    heap: list[tuple[int, RevBytes, RevBytes, int, int]],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> tuple[Counter[tuple[int, ...]], dict[tuple[int,int], int], dict[tuple[int,int], set[tuple[int,...]]],
     list[tuple[int, RevBytes, RevBytes, int, int]], dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 找出出现次数最多的 pair，若次数相同则选择 pair 最大的那个
    best_pair = None
    while heap:
        negc, ra, rb, a, b = heap[0]
        cur = pair_counts.get((a, b), 0)
        if cur <= 0 or -negc != cur:
            heapq.heappop(heap)
            continue
        best_pair = (a, b)
        break
    if not best_pair:
        return pretoken_counts, pair_counts, pair_sets, heap, vocab, merges

    # 更新 vocab 和 merges
    a, b = best_pair
    new_tok = vocab[a] + vocab[b]     # 新的 token
    new_id = len(vocab)
    vocab[new_id] = new_tok
    merges.append((vocab[a], vocab[b]))

    # 更新 pretoken_counts, pair_counts 和 pair_sets
    affected_seqs = list(pair_sets[best_pair])    # 所有包含 best_pair 的 seq
    touched_pairs: set[tuple[int, int]] = set()     # 所有出现次数发生变化的 pair
    for seq in affected_seqs:
        freq = pretoken_counts.pop(seq)
        seen_pairs = set()
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] -= freq  # 这个 pair 对总次数的贡献是 freq
            seen_pairs.add(pair)    # 先加入到 seen_pairs 中，防止重复删除
            touched_pairs.add(pair)
        for pair in seen_pairs:
            pair_sets[pair].remove(seq)     # pair 对应的集合中去掉 seq
            if pair_counts[pair] == 0:
                del pair_counts[pair]
                if not pair_sets[pair]:
                    del pair_sets[pair]

        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                new_seq.append(new_id)      # 可以合并，则添加合并后的 new_id，并且 i 要额外加一
                i += 2
            else:
                new_seq.append(seq[i])      # 否则添加原来的 id
                i += 1
        new_seq = tuple(new_seq)

        pretoken_counts[new_seq] += freq     # 累加新的 seq 出现的次数
        for i in range(len(new_seq) - 1):
            pair = (new_seq[i], new_seq[i + 1])
            pair_counts[pair] += freq   # 这个 pair 对总次数的贡献是 freq
            pair_sets[pair].add(new_seq)    # 新的 seq 加入 pair 对应的集合
            touched_pairs.add(pair)

    # 更新 heap
    for (x, y) in touched_pairs:
        c = pair_counts.get((x, y), 0)
        if c > 0:
            heapq.heappush(heap, (-c, RevBytes(vocab[x]), RevBytes(vocab[y]), x, y))

    return pretoken_counts, pair_counts, pair_sets, heap, vocab, merges

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
    special_id: dict[str, int] = {}

    for i in range(256):
        vocab[i] = bytes([i])   # 初始字符加入词表
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")  # 特殊字符加入词表
        special_id[special_tokens[i]] = i + 256

    pretoken_counts = pretokenize_file(input_path, special_tokens, special_id)      # 进行预分词

    # 计算每个 pair 出现的次数和每个 pair 分别在哪些 seq 中
    pair_counts = defaultdict(int)
    pair_sets = defaultdict(set)
    for seq, freq in pretoken_counts.items():  # 某个 pretoken 对应的整数序列和它出现的次数
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += freq  # 这个 pair 对总次数的贡献是 freq
            pair_sets[pair].add(seq)

    # 建立大根堆，用于后续取最大 pair
    heap: list[tuple[int, RevBytes, RevBytes, int, int]] = []
    for (a, b), c in pair_counts.items():
        if c > 0:
            heapq.heappush(heap, (-c, RevBytes(vocab[a]), RevBytes(vocab[b]), a, b))

    num_merges = vocab_size - 256 - len(special_tokens)
    for i in range(num_merges):
        pretoken_counts, pair_counts, pair_sets, heap, vocab, merges = merge(
            pretoken_counts, pair_counts, pair_sets, heap, vocab, merges
        )

    return vocab, merges

def train_bpe_from_counts(
    pretoken_counts: Counter[tuple[int, ...]],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train BPE starting from pretoken_counts (Counter of byte-id tuples).
    This lets external scripts do multiprocessing pretokenization and reuse our merge logic.
    """
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    for i in range(256):
        vocab[i] = bytes([i])
    for i, tok in enumerate(special_tokens):
        vocab[256 + i] = tok.encode("utf-8")

    pair_counts = defaultdict(int)
    pair_sets = defaultdict(set)
    for seq, freq in pretoken_counts.items():
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += freq
            pair_sets[pair].add(seq)

    heap: list[tuple[int, RevBytes, RevBytes, int, int]] = []
    for (a, b), c in pair_counts.items():
        if c > 0:
            heapq.heappush(heap, (-c, RevBytes(vocab[a]), RevBytes(vocab[b]), a, b))

    num_merges = vocab_size - 256 - len(special_tokens)
    for _ in range(num_merges):
        pretoken_counts, pair_counts, pair_sets, heap, vocab, merges = merge(
            pretoken_counts, pair_counts, pair_sets, heap, vocab, merges
        )

    return vocab, merges
