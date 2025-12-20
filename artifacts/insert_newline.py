def insert_newline(text: str) -> str:
    """Ensures a string ends with a newline character."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    if not text.endswith("\n"):
        return text + "\n"
    return text
