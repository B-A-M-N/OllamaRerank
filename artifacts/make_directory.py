import os


def make_directory(dir_path: str) -> None:
    """
    Creates a new directory.

    Args:
        dir_path (str): The path of the directory to create.

    Raises:
        OSError: If the directory cannot be created.
    """
    try:
        os.makedirs(dir_path)
    except OSError as err:
        raise err
