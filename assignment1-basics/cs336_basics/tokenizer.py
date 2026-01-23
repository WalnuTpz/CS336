from __future__ import annotations

import os
import base64
import json
from typing import Any, Generator

import regex as re
import heapq
from collections import Counter, defaultdict

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pretokenize_file(   # 对文本进行预分词
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

def merge(      # 单次合并操作
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

def train_bpe(      # 主训练函数
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

def train_bpe_from_counts(      # 给定 pretoken_counts 的训练函数
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
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")

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
    for i in range(num_merges):
        pretoken_counts, pair_counts, pair_sets, heap, vocab, merges = merge(
            pretoken_counts, pair_counts, pair_sets, heap, vocab, merges
        )

    return vocab, merges

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None
    ):
        self.id_to_bytes: dict[int, bytes] = dict(vocab)      # 浅拷贝，防止直接赋值时两者指向同一个 dict
        self.bytes_to_id: dict[bytes, int] = {}
        for i, b in self.id_to_bytes.items():
            self.bytes_to_id[b] = i
        self.byte_id:list[int] = []    # 单个 byte 的 int 值对应的 id
        for i in range(256):
            self.byte_id.append(self.bytes_to_id[bytes([i])])

        # 处理新添加的特殊 token
        self.special_tokens: list[str] = special_tokens if special_tokens else []
        self.special_set: set[str] = set(self.special_tokens)
        self.special_id: dict[str, int] = {}
        next_id = max(self.id_to_bytes.keys()) + 1
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b in self.bytes_to_id:
                b_id = self.bytes_to_id[b]      # 已在词表中的 special token 直接获取 id
            else:
                b_id = next_id      # 不在词表中的 special token 将 next_id 作为它的 id
                next_id += 1
                self.id_to_bytes[b_id] = b      # 将 token 和 id 加入词表和反向词表
                self.bytes_to_id[b] = b_id
            self.special_id[tok] = b_id
        if self.special_set:
            # 将 special set 排序，防止某个 special token 是另一个的子串时分割出错
            special_sorted = sorted(self.special_set, key = len, reverse = True)
            # 将排序后的 special token 修改为正则格式，用于后续对 text 进行分割
            delimit = "|".join(re.escape(tok) for tok in special_sorted)
            self._special_re = re.compile(f"({delimit})")
        else:
            self._special_re = None

        # 将 merges 中生成的所有新 token 的 rank,id 加入字典
        self.pair_rank: dict[tuple[int, int], int] = {}
        self.pair_id: dict[tuple[int, int], int] = {}
        for i in range(len(merges)):
            a, b = merges[i]
            id_pair = (self.bytes_to_id[a], self.bytes_to_id[b])
            new_tok = a + b
            new_id = self.bytes_to_id[new_tok]
            self.pair_rank[id_pair] = i
            self.pair_id[id_pair] = new_id

        self.bpe_cache: dict[bytes, tuple[int, ...]] = {}      # 用来储存 bytes 对应的 ids，便于后续复用

    @classmethod
    def from_files(     # 从文件中读取 vocab 和 merges 用来构造 Tokenizer 类
        cls,
        vocab_fp: str | os.PathLike,
        merges_fp: str | os.PathLike,
        special_tokens: list[str] = None,
    ) -> Tokenizer:
        # 读取 vocab.json
        with open(vocab_fp, "r", encoding = "utf-8") as f:
            vocab_raw = json.load(f)
        vocab:dict[int, bytes] = {}
        for k, v in vocab_raw.items():
            vocab[int(k)] = base64.b64decode(v)

        # 读取 merges.json
        with open(merges_fp, "r", encoding = "utf-8") as f:
            merges_raw = json.load(f)
        merges:list[tuple[bytes, bytes]] = []
        for item in merges_raw:
            a = base64.b64decode(item[0])
            b = base64.b64decode(item[1])
            merges.append((a, b))

        return cls(vocab, merges, special_tokens = special_tokens)

    def merge_ids(self, ids: list[int]) -> list[int]:   # 将 ids 中的元素按照训练规则合并
        while len(ids) >= 2:
            best_rank = float('inf')
            best_pair = None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = self.pair_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break

            new_id = self.pair_id[best_pair]
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i + 1] == best_pair[1]:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

        return ids

    def encode_token(self, tok: str) -> list[int]:    # 对单个 token 进行 encode
        b = tok.encode("utf-8")
        if b in self.bpe_cache:
            return list(self.bpe_cache[b])

        ids: list[int] = []
        for byte in b:     # 将 bytes 初始化为 ids
            ids.append(self.byte_id[byte])
        ids = self.merge_ids(ids)
        self.bpe_cache[b] = tuple(ids)
        return ids

    def encode(self, text: str):    # 将文本转换为编码
        if self.special_set:
            parts = self._special_re.split(text)
        else:
            parts = [text]

        ids:list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_set:
                ids.append(self.special_id[part])
            else:
                for m in PAT.finditer(part):
                    tok = m.group(0)
                    ids.extend(self.encode_token(tok))

        return ids

    def encode_iterable(self, iterable: list[str]) -> Generator[int, Any, None]:    # 将一个文本流进行 encode
        for s in iterable:
            for id_ in self.encode(s):
                yield id_      # 将每个 id 依次加入到生成器中

    def decode(self, ids: list[int]) -> str:    # 将编码转换为文本
        bytes_list: list[bytes] = []
        for id_ in ids:
            bytes_list.append(self.id_to_bytes[id_])
        bytes_text = b"".join(bytes_list)       # 将 bytes 列表合并为一整个 bytes 串
        text = bytes_text.decode("utf-8", errors = "replace")

        return text