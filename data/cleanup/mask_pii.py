import re

def mask_emails(text: str) -> tuple[str, int]:
    """
    Replace email addresses in text with a placeholder.

    Args:
        text (str): Input text.

    Returns:
        tuple[str, int]: A tuple of (masked text, number of replacements made).
    """
    PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    return re.subn(PATTERN, "|||EMAIL_ADDRESS|||", text)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    """
    Replace phone numbers in text with a placeholder.

    Args:
        text (str): Input text.

    Returns:
        tuple[str, int]: A tuple of (masked text, number of replacements made).
    """
    PATTERN = r"\(?\b(?:\+?\d{1,3}[-. ]?)?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"
    return re.subn(PATTERN, "|||PHONE_NUMBER|||", text)

def mask_ips(text: str) -> tuple[str, int]:
    """
    Replace IPv4 addresses in text with a placeholder.

    Args:
        text (str): Input text.

    Returns:
        tuple[str, int]: A tuple of (masked text, number of replacements made).
    """
    OCTET = r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
    PATTERN = rf"\b{OCTET}\.{OCTET}\.{OCTET}\.{OCTET}\b"

    return re.subn(PATTERN, "|||IP_ADDRESS|||", text)