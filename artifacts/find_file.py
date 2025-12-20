import os
import shutil


def find_file(path: str) -> str | None:
    """Finds a file at the specified path and returns its contents as a string. Returns None if no file is found."""
    if not os.path.exists(path):
        raise FileNotFoundError("The specified path does not exist.")

    if not os.path.isfile(path):
        raise NotADirectoryError("The specified path is a directory, not a file.")

    with open(path, "r") as file:
        contents = file.read()

    return contents if contents else None
