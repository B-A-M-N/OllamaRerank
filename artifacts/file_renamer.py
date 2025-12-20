import os


def file_renamer(file_path: str, new_filename: str) -> None:
    """
    Rename a file.

    Args:
        file_path (str): The path of the file to rename.
        new_filename (str): The new name for the file.

    Returns:
        None.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File '{file_path}' does not exist.")

    new_file_path = os.path.join(os.path.dirname(file_path), new_filename)

    try:
        os.rename(file_path, new_file_path)
    except OSError as e:
        raise PermissionError(
            f"Unable to rename file '{file_path}' to '{new_file_path}': {e}"
        )

    return None
