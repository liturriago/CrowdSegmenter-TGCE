def format_time(seconds: float) -> str:
    """Converts seconds into a readable MM:SS format.

    Args:
        seconds (float): Total seconds to format.

    Returns:
        str: Formatted time string (MM:SS).
    """
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"