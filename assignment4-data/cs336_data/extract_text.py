from __future__ import annotations

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def decode_html_bytes(html_bytes: bytes) -> str:    # 将 HTML bytes 解码为字符串
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


def extract_text_from_html_bytes(html_bytes: bytes) -> str:    # 从 HTML bytes 中抽取纯文本
    html_str = decode_html_bytes(html_bytes)    
    text = extract_plain_text(html_str)   

    return text