from __future__ import annotations

import os
from typing import Any


from cs336_data.extract_text import extract_text_from_html_bytes

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_html_bytes(html_bytes)

from cs336_data.language_identification import identify_language

def run_identify_language(text: str) -> tuple[Any, float]:
    return identify_language(text)


from cs336_data.mask_pii import mask_emails, mask_phone_numbers, mask_ips

def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)



def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ips(text)

from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech

def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return classify_toxic_speech(text)



from cs336_data.quality_classifier import classify_quality

def run_classify_quality(text: str) -> tuple[Any, float]:
    return classify_quality(text)


from cs336_data.gopher_quality import gopher_quality_filter

def run_gopher_quality_filter(text: str) -> bool:
    return gopher_quality_filter(text)


from cs336_data.exact_deduplication import exact_line_deduplication
from cs336_data.minhash_deduplication import minhash_deduplication

def run_exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    return exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return minhash_deduplication(
        input_files=input_files,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        jaccard_threshold=jaccard_threshold,
        output_directory=output_directory,
    )
