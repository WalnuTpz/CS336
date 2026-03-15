from __future__ import annotations

from pathlib import Path
from typing import Tuple

import fasttext

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments/7.Problem quality_classifier/models/quality_fasttext.bin"
)
_QUALITY_MODEL = None


def _normalized_text(text: str) -> str:    # 去掉换行、制表符和多余空格
    return " ".join(text.split())


def get_quality_model():    # 加载 fastText 质量分类模型
    global _QUALITY_MODEL

    if _QUALITY_MODEL is None:
        _QUALITY_MODEL = fasttext.load_model(str(MODEL_PATH))

    return _QUALITY_MODEL


def classify_quality(text: str) -> Tuple[str, float]:    # 质量分类函数
    cleaned = _normalized_text(text)
    model = get_quality_model()

    labels, scores = model.predict(cleaned, k=1)
    label = labels[0].removeprefix("__label__")    # 去掉 fastText 的标签前缀
    score = float(scores[0])
    
    return label, score
