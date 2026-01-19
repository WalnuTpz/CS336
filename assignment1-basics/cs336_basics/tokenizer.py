from __future__ import annotations

import os
from typing import Tuple

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
    raise NotImplementedError
