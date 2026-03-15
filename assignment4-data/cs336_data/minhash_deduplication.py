from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import os
import re
import unicodedata
import mmh3


MAX_HASH = (1 << 64) - 1
TOKEN_RE = re.compile(r"\w+")


def get_ngrams(text: str, ngrams: int) -> set[str]:    # 取词级 n-gram；太短时退化成整个文本
    # 统一大小写、重音和空白
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = " ".join(text.lower().split())

    tokens = TOKEN_RE.findall(text)    # 把文本切成词

    if not tokens:    # 没有词时直接返回空集合
        return set()
    if len(tokens) < ngrams:    # 词数不足时直接把所有词当成一个 n-gram
        return {" ".join(tokens)}
    
    ngram_set = {" ".join(tokens[i : i + ngrams]) for i in range(len(tokens) - ngrams + 1)}

    return ngram_set


def compute_minhash_signature(ngram_set: set[str], num_hashes: int) -> tuple[int, ...]:    # 给每个文档算 minhash 签名
    if not ngram_set:
        return tuple(MAX_HASH for _ in range(num_hashes))

    signature: list[int] = []
    for seed in range(num_hashes):    # 使用不同的 seed 来模拟不同的哈希函数
        min_hash = min(mmh3.hash64(ngram, seed=seed, signed=False)[0] for ngram in ngram_set)
        signature.append(min_hash)

    return tuple(signature)


def get_band_buckets(    # 按 band 切分签名，做 LSH 分桶
    signatures: list[tuple[int, ...]],
    num_bands: int,
) -> dict[tuple[int, tuple[int, ...]], list[int]]:
    rows_per_band = len(signatures[0]) // num_bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)

    for doc_idx, signature in enumerate(signatures):    # 遍历每个文档的编号和签名
        for band_idx in range(num_bands):    # 将签名切成 num_bands 个 band
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band = signature[start:end]
            buckets[(band_idx, band)].append(doc_idx)

    return buckets


def collect_candidate_pairs(    # 同桶的文档两两组成候选对
    buckets: dict[tuple[int, tuple[int, ...]], list[int]],
) -> set[tuple[int, int]]:
    candidate_pairs: set[tuple[int, int]] = set()

    for doc_indices in buckets.values():
        if len(doc_indices) < 2:
            continue
        for i in range(len(doc_indices)):
            for j in range(i + 1, len(doc_indices)):
                left = doc_indices[i]
                right = doc_indices[j]
                if left > right:
                    left, right = right, left
                candidate_pairs.add((left, right))

    return candidate_pairs


def jaccard_similarity(left: set[str], right: set[str]) -> float:    # 用真实 Jaccard 复核候选对
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    
    intersection = len(left & right)
    union = len(left | right)
    similarity = intersection / union

    return similarity


def find(parent: list[int], x: int) -> int:    # 并查集查找
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent: list[int], x: int, y: int) -> None:    # 并查集合并
    root_x = find(parent, x)
    root_y = find(parent, y)
    if root_x != root_y:
        parent[root_y] = root_x


def choose_kept_indices(num_docs: int, duplicate_pairs: set[tuple[int, int]]) -> set[int]:    # 每个重复簇只保留输入顺序最早的文档
    parent = list(range(num_docs))

    for left, right in duplicate_pairs:
        union(parent, left, right)

    clusters: dict[int, list[int]] = defaultdict(list)    # 将每个文档添加到其所在的重复簇中
    for doc_idx in range(num_docs):
        clusters[find(parent, doc_idx)].append(doc_idx)

    kept_indices: set[int] = set()
    for cluster in clusters.values():
        kept_indices.add(min(cluster))

    return kept_indices


def minhash_deduplication(    # 主函数：做 minhash 去重并把保留的原文件写出去
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    input_paths = [Path(path) for path in input_files]
    documents: list[str] = []
    ngram_sets: list[set[str]] = []

    for input_path in input_paths:
        text = input_path.read_text(encoding="utf-8")
        documents.append(text)    # 保留文档原文以便最后写出
        ngram_sets.append(get_ngrams(text, ngrams))    

    signatures = [compute_minhash_signature(ngram_set, num_hashes) for ngram_set in ngram_sets]
    buckets = get_band_buckets(signatures, num_bands)
    candidate_pairs = collect_candidate_pairs(buckets)

    duplicate_pairs: set[tuple[int, int]] = set()
    for left, right in candidate_pairs:    # 计算候选对的 Jaccard 相似度，找出真正的重复对
        similarity = jaccard_similarity(ngram_sets[left], ngram_sets[right])
        if similarity >= jaccard_threshold:
            duplicate_pairs.add((left, right))

    kept_indices = choose_kept_indices(len(input_paths), duplicate_pairs)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for doc_idx, input_path in enumerate(input_paths):    # 将保留的文档写出到输出目录，文件名不变
        if doc_idx not in kept_indices:
            continue
        output_path = output_directory / input_path.name
        output_path.write_text(documents[doc_idx], encoding="utf-8")
