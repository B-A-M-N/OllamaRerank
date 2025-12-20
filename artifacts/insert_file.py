import os
import shutil


def insert_file(source: str, destination: str) -> None:
    """
    Inserts a file at the specified destination path.

    :param source: The path to the file to be inserted.
    :type source: str
    :param destination: The path where the file will be inserted.
    :type destination: str
    :raises FileNotFoundError: If the source file does not exist.
    :raises FileExistsError: If the destination path already exists and cannot be overwritten.
    """
    if not os.path.exists(source):
        raise FileNotFoundError("Source file does not exist.")

    # Check if the destination directory exists, create it if necessary
    dest_dir = os.path.dirname(destination)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    # Move the file to the specified destination
    shutil.move(source, destination)
