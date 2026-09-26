def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a short "1h23m" / "4m05s" / "12s" string.

    Args:
        seconds (float): Duration in seconds.
    Returns:
        str: Human-readable duration.
    """
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    
    return f"{secs}s"
