from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import os


def hash_line(line: str) -> str:    # 对一整行做哈希
    return hashlib.sha1(line.encode("utf-8")).hexdigest()


def count_line_hashes(input_files: list[os.PathLike]) -> Counter[str]:    # 统计每一行（的哈希）出现了多少次
    counts: Counter[str] = Counter()

    for input_path in input_files:
        input_path = Path(input_path)
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                h = hash_line(line)
                counts[h] += 1

    return counts


def rewrite_deduplicated_files(    # 重写文件，只保留全局唯一行
    input_files: list[os.PathLike],
    output_directory: os.PathLike,
    counts: Counter[str],
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for input_path in input_files:
        input_path = Path(input_path)
        output_path = output_directory / input_path.name

        with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                h = hash_line(line)
                if counts[h] == 1:
                    fout.write(line)


def exact_line_deduplication(    # 主函数
    input_files: list[os.PathLike],
    output_directory: os.PathLike,
) -> None:
    counts = count_line_hashes(input_files)
    rewrite_deduplicated_files(input_files, output_directory, counts)