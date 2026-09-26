from functools import lru_cache

import fasttext
from fasttext.FastText import _FastText

from config import DataConfig

@lru_cache(maxsize=1)
def load_quality_model() -> _FastText:
    """
    Load and cache the trained quality classifier model.

    Returns:
        _FastText: The loaded fastText model.
    """
    return fasttext.load_model(DataConfig.quality_clf_model_path)

def transform_to_singleline(text: str) -> str:
    """
    Collapse carriage returns, newlines, and tabs into single spaces.

    Args:
        text (str): Input text.

    Returns:
        str: Text with line breaks and tabs replaced by spaces.
    """
    return text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

def classify_quality(text: str) -> tuple[str, float]:
    """
    Classify text as high or low quality using the trained classifier.

    Args:
        text (str): Input text.

    Returns:
        tuple[str, float]: A tuple of (predicted label, confidence score).
    """
    model = load_quality_model()
    prediction = model.predict(transform_to_singleline(text))

    return (prediction[0][0].replace("__label__", ""), prediction[1][0])