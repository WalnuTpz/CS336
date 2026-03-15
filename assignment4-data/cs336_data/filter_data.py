from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from fastwarc.stream_io import FileStream, GZipStream
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tldextract import TLDExtract
from xopen import xopen

from cs336_data.gopher_quality import gopher_quality_filter
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.language_identification import identify_language
from cs336_data.mask_pii import mask_emails, mask_ips, mask_phone_numbers


DEFAULT_MIN_CHARS = 200
DEFAULT_MIN_WORDS = 50
DEFAULT_LANG_SCORE_THRESHOLD = 0.80
DEFAULT_NSFW_SCORE_THRESHOLD = 0.95
DEFAULT_TOXIC_SCORE_THRESHOLD = 0.98
DEFAULT_MAX_DOCS_PER_DOMAIN = 50
BLOCKED_TOPIC_KEYWORDS = {
    "casino",
    "casinos",
    "betting",
    "sportsbook",
    "poker",
    "slots",
    "blackjack",
    "roulette",
    "jackpot",
    "gambling",
}
PARKED_DOMAIN_PHRASES = {
    "domain has expired",
    "this domain has expired",
    "under suspension",
    "renew this domain name",
    "permanently removed from your account",
    "buy this domain",
    "parked free courtesy of",
}
NAVIGATION_PHRASES = {
    "privacy policy",
    "terms of service",
    "all rights reserved",
    "skip to content",
    "shopping cart",
    "sign in",
    "create account",
    "contact us",
    "log in",
    "register",
}


def normalize_text(text: str) -> str:    # 去掉换行、制表符和多余空格，输出单行文档
    return " ".join(text.split())


def count_words(text: str) -> int:    # 粗略词数统计
    return len(text.split())


def hash_document(text: str) -> str:    # 对归一化后的文档做哈希，用于局部 exact dedup
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def passes_blocked_topic_filter(uri: str, text: str) -> bool:    # 保守拦截明显博彩页
    lowered = f"{uri} {text[:2000]}".lower()
    hits = sum(1 for keyword in BLOCKED_TOPIC_KEYWORDS if keyword in lowered)
    return hits < 2


def passes_parked_domain_filter(text: str) -> bool:    # 拦截过期域名和停放页
    lowered = text.lower()
    return not any(phrase in lowered for phrase in PARKED_DOMAIN_PHRASES)


def passes_boilerplate_filter(raw_text: str, normalized_text: str) -> bool:    # 拦截模板和导航占比过高的页面
    normalized_lines = [normalize_text(line).lower() for line in raw_text.splitlines()]
    normalized_lines = [line for line in normalized_lines if line]
    if len(normalized_lines) >= 5:
        duplicate_line_ratio = 1 - len(set(normalized_lines)) / len(normalized_lines)
        if duplicate_line_ratio > 0.20:
            return False

    words = normalized_text.lower().split()
    if not words:
        return False

    nav_hits = sum(normalized_text.lower().count(phrase) for phrase in NAVIGATION_PHRASES)
    if len(words) >= 80 and nav_hits / len(words) > 0.03:
        return False

    if len(words) >= 200:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.25:
            return False

    return True


def build_output_path(input_path: Path, output_directory: Path) -> Path:    # 给每个 WET 文件生成对应输出路径
    stem = input_path.name
    if stem.endswith(".gz"):
        stem = stem[:-3]
    return output_directory / f"{stem}.filtered.txt.gz"


def build_stats_path(input_path: Path, output_directory: Path) -> Path:    # 给每个 WET 文件生成对应统计路径
    stem = input_path.name
    if stem.endswith(".gz"):
        stem = stem[:-3]
    return output_directory / f"{stem}.stats.json"


def make_tld_extractor() -> TLDExtract:    # 用内置 suffix snapshot，避免运行时联网拉 PSL
    return TLDExtract(suffix_list_urls=(), cache_dir=None)


def get_registered_domain(uri: str, extractor: TLDExtract) -> str:    # 提取 eTLD+1，失败时退回 netloc
    extracted = extractor(uri)
    if extracted.top_domain_under_public_suffix:
        return extracted.top_domain_under_public_suffix
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    return extracted.domain or extracted.fqdn or uri


def mask_all_pii(text: str) -> tuple[str, dict[str, int]]:    # 统一做 PII masking，并统计替换次数
    masked_text, email_count = mask_emails(text)
    masked_text, phone_count = mask_phone_numbers(masked_text)
    masked_text, ip_count = mask_ips(masked_text)

    counts = {
        "emails": email_count,
        "phones": phone_count,
        "ips": ip_count,
    }
    return masked_text, counts


def is_conservatively_harmful(    # 只在分数很高时才丢弃，尽量减少误杀
    text: str,
    nsfw_score_threshold: float,
    toxic_score_threshold: float,
) -> tuple[bool, str | None]:
    nsfw_label, nsfw_score = classify_nsfw(text)
    if nsfw_label != "non-nsfw" and nsfw_score >= nsfw_score_threshold:
        return True, "nsfw"

    toxic_label, toxic_score = classify_toxic_speech(text)
    if toxic_label != "non-toxic" and toxic_score >= toxic_score_threshold:
        return True, "toxic"

    return False, None


def make_stats(input_path: Path) -> dict[str, Any]:    # 初始化单文件统计
    return {
        "input_file": str(input_path),
        "output_file": "",
        "input_records": 0,
        "kept_records": 0,
        "first_failure_counts": {
            "missing_uri": 0,
            "empty_text": 0,
            "too_short": 0,
            "blocked_topic": 0,
            "parked_domain": 0,
            "non_english": 0,
            "low_gopher_quality": 0,
            "boilerplate": 0,
            "nsfw": 0,
            "toxic": 0,
            "exact_duplicate": 0,
            "domain_cap": 0,
        },
        "pii_docs_modified": 0,
        "pii_replacements": {
            "emails": 0,
            "phones": 0,
            "ips": 0,
        },
        "domain_counts_kept": {},
    }


def increment_failure(stats: dict[str, Any], reason: str) -> None:    # 记录文档第一次被过滤掉的原因
    stats["first_failure_counts"][reason] += 1


def merge_stats(all_stats: list[dict[str, Any]]) -> dict[str, Any]:    # 汇总多文件统计
    summary = {
        "num_input_files": len(all_stats),
        "input_records": 0,
        "kept_records": 0,
        "first_failure_counts": Counter(),
        "pii_docs_modified": 0,
        "pii_replacements": Counter(),
        "input_files": [],
        "output_files": [],
    }

    for stats in all_stats:
        summary["input_records"] += stats["input_records"]
        summary["kept_records"] += stats["kept_records"]
        summary["pii_docs_modified"] += stats["pii_docs_modified"]
        summary["input_files"].append(stats["input_file"])
        summary["output_files"].append(stats["output_file"])
        summary["first_failure_counts"].update(stats["first_failure_counts"])
        summary["pii_replacements"].update(stats["pii_replacements"])

    summary["first_failure_counts"] = dict(summary["first_failure_counts"])
    summary["pii_replacements"] = dict(summary["pii_replacements"])
    return summary


def process_single_wet_file(    # 处理单个 WET 文件：过滤、mask、写出文本和统计
    input_path: str,
    output_directory: str,
    min_chars: int,
    min_words: int,
    lang_score_threshold: float,
    nsfw_score_threshold: float,
    toxic_score_threshold: float,
    max_docs_per_domain: int,
    max_records_per_file: int | None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = build_output_path(input_path, output_directory)
    stats_path = build_stats_path(input_path, output_directory)
    stats = make_stats(input_path)
    stats["output_file"] = str(output_path)

    extractor = make_tld_extractor()
    seen_hashes: set[str] = set()
    domain_counts: Counter[str] = Counter()

    stream = GZipStream(FileStream(str(input_path), "rb"))
    with xopen(output_path, "wt", encoding="utf-8") as fout:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):
            if max_records_per_file is not None and stats["input_records"] >= max_records_per_file:
                break
            stats["input_records"] += 1

            uri = record.headers.get("WARC-Target-URI")
            if not uri:
                increment_failure(stats, "missing_uri")
                continue

            raw_text = bytes(record.reader.read()).decode("utf-8", errors="replace")
            normalized = normalize_text(raw_text)
            if not normalized:
                increment_failure(stats, "empty_text")
                continue

            if len(normalized) < min_chars or count_words(normalized) < min_words:
                increment_failure(stats, "too_short")
                continue

            if not passes_blocked_topic_filter(uri, normalized):
                increment_failure(stats, "blocked_topic")
                continue

            if not passes_parked_domain_filter(normalized):
                increment_failure(stats, "parked_domain")
                continue

            language, language_score = identify_language(normalized)
            if language != "en" or language_score < lang_score_threshold:
                increment_failure(stats, "non_english")
                continue

            if not gopher_quality_filter(normalized):
                increment_failure(stats, "low_gopher_quality")
                continue

            if not passes_boilerplate_filter(raw_text, normalized):
                increment_failure(stats, "boilerplate")
                continue

            is_harmful, harmful_reason = is_conservatively_harmful(
                normalized,
                nsfw_score_threshold=nsfw_score_threshold,
                toxic_score_threshold=toxic_score_threshold,
            )
            if is_harmful:
                increment_failure(stats, harmful_reason or "toxic")
                continue

            masked_text, pii_counts = mask_all_pii(normalized)
            if sum(pii_counts.values()) > 0:
                stats["pii_docs_modified"] += 1
            for key, value in pii_counts.items():
                stats["pii_replacements"][key] += value

            doc_hash = hash_document(masked_text)
            if doc_hash in seen_hashes:
                increment_failure(stats, "exact_duplicate")
                continue
            seen_hashes.add(doc_hash)

            domain = get_registered_domain(uri, extractor)
            if domain_counts[domain] >= max_docs_per_domain:
                increment_failure(stats, "domain_cap")
                continue
            domain_counts[domain] += 1

            fout.write(masked_text)
            fout.write("\n")
            stats["kept_records"] += 1

    stats["domain_counts_kept"] = dict(domain_counts)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def expand_input_paths(inputs: list[str]) -> list[str]:    # 展开 glob，并去重排序
    input_paths: set[str] = set()

    for item in inputs:
        matches = glob.glob(item)
        if matches:
            input_paths.update(matches)
        else:
            input_paths.add(item)

    return sorted(input_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Common Crawl WET files for LM training.")
    parser.add_argument("inputs", nargs="+", help="WET files or glob patterns, e.g. '/data/CC/CC*.warc.wet.gz'")
    parser.add_argument("--output-dir", required=True, help="Directory for filtered text and stats")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of worker processes to use")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--lang-score-threshold", type=float, default=DEFAULT_LANG_SCORE_THRESHOLD)
    parser.add_argument("--nsfw-score-threshold", type=float, default=DEFAULT_NSFW_SCORE_THRESHOLD)
    parser.add_argument("--toxic-score-threshold", type=float, default=DEFAULT_TOXIC_SCORE_THRESHOLD)
    parser.add_argument("--max-docs-per-domain", type=int, default=DEFAULT_MAX_DOCS_PER_DOMAIN)
    parser.add_argument("--max-records-per-file", type=int, default=None, help="Optional cap for quick smoke tests")
    return parser.parse_args()


def main() -> None:    # 命令行入口：并行处理多个 WET 文件，并写总统计
    args = parse_args()
    input_paths = expand_input_paths(args.inputs)
    if not input_paths:
        raise ValueError("No input WET files matched the provided patterns.")

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    worker_count = max(1, args.max_workers)
    all_stats: list[dict[str, Any]] = []

    if worker_count == 1:
        for index, input_path in enumerate(input_paths, 1):
            print(
                f"[filter] {index}/{len(input_paths)} {input_path}",
                file=sys.stderr,
            )
            stats = process_single_wet_file(
                input_path=input_path,
                output_directory=str(output_directory),
                min_chars=args.min_chars,
                min_words=args.min_words,
                lang_score_threshold=args.lang_score_threshold,
                nsfw_score_threshold=args.nsfw_score_threshold,
                toxic_score_threshold=args.toxic_score_threshold,
                max_docs_per_domain=args.max_docs_per_domain,
                max_records_per_file=args.max_records_per_file,
            )
            all_stats.append(stats)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for input_path in input_paths:
                future = executor.submit(
                    process_single_wet_file,
                    input_path,
                    str(output_directory),
                    args.min_chars,
                    args.min_words,
                    args.lang_score_threshold,
                    args.nsfw_score_threshold,
                    args.toxic_score_threshold,
                    args.max_docs_per_domain,
                    args.max_records_per_file,
                )
                futures.append((input_path, future))

            for index, (input_path, future) in enumerate(futures, 1):
                stats = future.result()
                print(
                    f"[done] {index}/{len(input_paths)} {input_path} kept={stats['kept_records']}",
                    file=sys.stderr,
                )
                all_stats.append(stats)

    summary = merge_stats(all_stats)
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
