import os


def create_folder(path: str) -> None:
    """Creates a new folder at the specified path."""
    try:
        os.mkdir(path)
    except FileExistsError:
        raise ValueError(f"Folder already exists at {{path}}")
