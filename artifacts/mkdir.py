import os


def mkdir(folder_name: str) -> None:
    """
    Creates a new directory named `folder_name` if it does not already exist.

    Raises:
        FileExistsError: If the specified folder already exists.
        IOError: If an I/O error occurs during the creation of the directory.
        PermissionError: If the user does not have permission to create the directory.
    """
    try:
        os.makedirs(folder_name)
    except Exception as e:
        raise
