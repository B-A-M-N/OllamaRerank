import os
import shutil


def duplicate_file(src_path: str, dst_path: str) -> None:
    # Check if the source file exists
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file does not exist at {src_path}")

    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Duplicate the file
    shutil.copy2(src_path, dst_path)
