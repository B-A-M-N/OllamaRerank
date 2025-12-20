import os
import shutil


def copy_file(src: str, dst: str) -> None:
    """Copies the contents of one file to another.

    Args:
        src (str): Path to the source file.
        dst (str): Path to the destination file.

    Raises:
        FileNotFoundError: If the source file does not exist.
        FileExistsError: If the destination file already exists and overwrite is not allowed.
    """
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source file {src} does not exist.")
    if os.path.exists(dst) and not os.access(dst, os.W_OK):
        raise FileExistsError(f"Destination file {dst} already exists.")
    shutil.copyfile(src, dst)
