import os
import shutil


def rename_file(src_path: str, dst_path: str) -> None:
    """Renames a file from `src_path` to `dst_path`. If the source does not exist, raises FileNotFoundError. If the destination already exists, raises FileExistsError."""

    # Check if source path exists
    if not os.path.exists(src_path):
        raise FileNotFoundError("The source file does not exist.")

    # Check if destination path already exists
    if os.path.exists(dst_path):
        raise FileExistsError("The destination path already exists.")

    # Rename the file from src_path to dst_path
    os.rename(src_path, dst_path)
