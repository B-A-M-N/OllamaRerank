def truncate_text(text: str, max_chars: int) -> str:
    """
    Hard truncates the given text to a maximum number of characters.
    This is a placeholder for a more sophisticated token-based truncation
    that would use a model's tokenizer.
    """
    if len(text) > max_chars:
        return text[:max_chars]
    return text
