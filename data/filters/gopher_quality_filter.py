from config import DataConfig

def gopher_quality_filter(text: str) -> bool:
    """
    Apply Gopher-style heuristic quality filters to a document.

    Args:
        text (str): Document text.

    Returns:
        bool:True if the document passes all quality heuristics, False otherwise.
    """
    # Length filter
    words = text.split()
    if len(words) < DataConfig.doc_min_length or len(words) > DataConfig.doc_max_length:
        return False

    # Mean-word-length and alphabetic-word checks
    total_word_len = 0
    alpha_word_count = 0
    for word in words:
        total_word_len += len(word)
        if any(char.isalpha() for char in word):
            alpha_word_count += 1

    mean_length = total_word_len / len(words)
    if mean_length < DataConfig.doc_min_mean_length or mean_length > DataConfig.doc_max_mean_length:
        return False

    if (alpha_word_count / len(words)) * 100 < 80:
        return False

    # Ellipsis ending filter
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    ellipsis_count = sum(1 for line in lines if line.endswith("..."))
    if (ellipsis_count / len(lines)) * 100 > 30:
        return False

    return True