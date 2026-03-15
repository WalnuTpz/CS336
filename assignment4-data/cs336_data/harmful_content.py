from __future__ import annotations

from pathlib import Path
from typing import Tuple

import fasttext

NSFW_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "nsfw_model.bin"
TOXIC_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "hatespeech_model.bin"

_NSFW_MODEL = None
_TOXIC_MODEL = None


def get_nsfw_model():
    global _NSFW_MODEL

    if _NSFW_MODEL is None:
        _NSFW_MODEL = fasttext.load_model(str(NSFW_MODEL_PATH))

    return _NSFW_MODEL


def get_toxic_model():
    global _TOXIC_MODEL

    if _TOXIC_MODEL is None:
        _TOXIC_MODEL = fasttext.load_model(str(TOXIC_MODEL_PATH))

    return _TOXIC_MODEL


def classify_nsfw(text: str) -> Tuple[str, float]:
    model = get_nsfw_model()
    cleaned = " ".join(text.split())   # 去掉换行、制表符和多余空格

    labels, scores = model.predict(cleaned, k=1)
    label = labels[0].removeprefix("__label__")
    score = float(scores[0])

    return label, score


def classify_toxic_speech(text: str) -> Tuple[str, float]:
    model = get_toxic_model()
    cleaned = " ".join(text.split())   

    labels, scores = model.predict(cleaned, k=1)
    label = labels[0].removeprefix("__label__")
    score = float(scores[0])

    return label, score