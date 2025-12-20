import os
import shutil


def move_file(src_path: str, dst_path: str) -> None:
    """
    Moves a file from a source path to a destination path.

    Args:
        src_path (str): The path to the source file.
        dst_path (str): The path to the destination file.

    Raises:
        FileNotFoundError: If the source file does not exist.
        FileExistsError: If the destination file already exists.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError("Source file does not exist.")

    if os.path.exists(dst_path) and not os.path.isdir(dst_path):
        raise FileExistsError("Destination file already exists.")

    shutil.move(src_path, dst_path)
