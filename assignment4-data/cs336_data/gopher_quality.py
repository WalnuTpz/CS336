from __future__ import annotations

from nltk.tokenize import word_tokenize


def get_words(text: str) -> list[str]:    # 从文本中提取词列表，保留行结构以便后续分析
    text = text.strip()
    if not text:
        return []
    
    return word_tokenize(text, preserve_line=True)


def passes_word_count(words: list[str]) -> bool:    # 词数必须在 [50, 100000] 之间
    n = len(words)

    return 50 <= n <= 100000


def passes_mean_word_length(words: list[str]) -> bool:    # 平均词长必须在 [3, 10] 之间
    if not words:
        return False

    total_chars = sum(len(w) for w in words)
    mean_len = total_chars / len(words)

    return 3 <= mean_len <= 10


def passes_ellipsis_line_ratio(text: str) -> bool:    # 以 '...' 结尾的行占比不能超过 30%
    lines = text.splitlines()

    nonempty_lines = [line.strip() for line in lines if line.strip()]    # 忽略空行
    if not nonempty_lines:
        return False

    ellipsis_lines = sum(1 for line in nonempty_lines if line.endswith("..."))
    ratio = ellipsis_lines / len(nonempty_lines)

    return ratio <= 0.30


def passes_alpha_word_ratio(words: list[str]) -> bool:    # 至少 80% 的词含有一个字母字符
    if not words:
        return False

    alpha_words = sum(1 for w in words if any(ch.isalpha() for ch in w))
    ratio = alpha_words / len(words)
    return ratio >= 0.80


def gopher_quality_filter(text: str) -> bool:    # 判断文本是否满足 Gopher 质量标准
    words = get_words(text)

    if not passes_word_count(words):
        return False
    if not passes_mean_word_length(words):
        return False
    if not passes_ellipsis_line_ratio(text):
        return False
    if not passes_alpha_word_ratio(words):
        return False

    return True