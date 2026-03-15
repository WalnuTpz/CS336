from __future__ import annotations

import re
from typing import Tuple

EMAIL_RE = re.compile(
    r"""
    \b
    [A-Za-z0-9._%+-]+
    @
    [A-Za-z0-9.-]+
    \.[A-Za-z]{2,}
    \b
    """,
    re.VERBOSE,
)

PHONE_RE = re.compile(
    r"""
    (?<!\d)
    (?:\(\d{3}\)|\d{3})
    [ -]?
    \d{3}
    [ -]?
    \d{4}
    (?!\d)
    """,
    re.VERBOSE,
)

OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"    # [0, 255]
IPV4_RE = re.compile(
    rf"""
    (?<!\d)
    {OCTET}\.{OCTET}\.{OCTET}\.{OCTET}
    (?!\d)
    """,
    re.VERBOSE,
)


def _sub_and_count(    # 通用替换函数
    pattern: re.Pattern, 
    replacement: str, 
    text: str
    ) -> Tuple[str, int]:
    new_text, count = pattern.subn(replacement, text)

    return new_text, count


def mask_emails(text: str) -> Tuple[str, int]:    # 遮蔽所有电子邮件地址
    return _sub_and_count(EMAIL_RE, "|||EMAIL_ADDRESS|||", text)


def mask_phone_numbers(text: str) -> Tuple[str, int]:    # 遮蔽所有电话号码
    return _sub_and_count(PHONE_RE, "|||PHONE_NUMBER|||", text)


def mask_ips(text: str) -> Tuple[str, int]:    # 遮蔽所有 IPv4 地址
    return _sub_and_count(IPV4_RE, "|||IP_ADDRESS|||", text)