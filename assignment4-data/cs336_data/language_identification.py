from __future__ import annotations

from pathlib import Path
from typing import Tuple

import fasttext

MODEL_PATH = Path(__file__).resolve().parent / "assets" / "lid.176.bin"
_LANG_MODEL = None


def get_language_model():    # 加载 fastText 语言识别模型。
    global _LANG_MODEL

    if _LANG_MODEL is None:
        _LANG_MODEL = fasttext.load_model(str(MODEL_PATH))

    return _LANG_MODEL


def identify_language(text: str) -> Tuple[str, float]:    # 语言识别函数
    model = get_language_model()
    cleaned = " ".join(text.split())   # 去掉换行、制表符和多余空格

    labels, scores = model.predict(cleaned, k=1)
    lang = labels[0].removeprefix("__label__")
    score = float(scores[0])

    return lang, score