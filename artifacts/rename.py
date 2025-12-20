import os
from pathlib import Path


def rename(old_name: str, new_name: str) -> None:
    if not os.path.exists(old_name):
        raise FileNotFoundError(f"File {old_name} does not exist.")
    if os.path.exists(new_name):
        raise ValueError(f"File {new_name} already exists.")
    try:
        os.rename(
            os.path.join(os.getcwd(), old_name), os.path.join(os.getcwd(), new_name)
        )
    except OSError as e:
        raise PermissionError(f"Cannot rename file {old_name} to {new_name}.") from e
