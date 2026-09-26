from functools import lru_cache

import fasttext
from fasttext.FastText import _FastText

from config import DataConfig

@lru_cache(maxsize=1)
def load_fasttext_model() -> _FastText:
    """
    Load and cache the fastText language-identification model.

    Returns:
        _FastText: The loaded fastText model.
    """
    return fasttext.load_model(DataConfig.language_clf_model_path)

def transform_to_singleline(text: str) -> str:
    """
    Collapse carriage returns, newlines, and tabs into single spaces.

    Args:
        text (str): Input text.

    Returns:
        str: Text with line breaks and tabs replaced by spaces.
    """
    return text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

def identify_language(text: str) -> tuple[str, float]:
    """
    Predict the most likely language of a piece of text.

    Args:
        text (str): Input text.

    Returns:
        tuple[str, float]: A tuple of (language code, confidence score).
    """
    model = load_fasttext_model()

    predictions = model.predict(transform_to_singleline(text), k=1)
    return (predictions[0][0].replace("__label__", ""), predictions[1][0])