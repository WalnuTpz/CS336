from __future__ import annotations

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def decode_html_bytes(html_bytes: bytes) -> str:
    """
    将原始 HTML bytes 解码为 Python 字符串。
    先尝试 UTF-8；如果失败，再做编码检测。
    """
    try:
        text = html_bytes.decode("utf-8")
        return text
    except UnicodeDecodeError:
        pass

    encoding = detect_encoding(html_bytes)
    try:
        return html_bytes.decode(encoding, errors="replace")
    except (LookupError, UnicodeDecodeError, TypeError):
        pass

    return html_bytes.decode("utf-8", errors="replace")


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """
    输入：原始 HTML bytes
    输出：提取出的纯文本
    """
    html_str = decode_html_bytes(html_bytes)    # 把 bytes 解码成 str
    text = extract_plain_text(html_str)    # 从 HTML str 中抽取文本

    return text