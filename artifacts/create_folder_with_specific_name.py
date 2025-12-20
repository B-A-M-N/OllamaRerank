import os


def create_folder_with_specific_name(folder_path: str, exist_ok: bool = True) -> None:
    """
    Creates a folder at the specified path. If the folder already exists, an error will be raised.

    :param folder_path: The path where the folder should be created.
    :param exist_ok: If False, FileExistsError is raised if the target directory already exists.
    :raises FileExistsError: Raised if the folder already exists and exist_ok is False.
    """
    if not isinstance(folder_path, str):
        raise TypeError("folder_path must be a string")
        
    try:
        os.makedirs(folder_path, exist_ok=exist_ok)
    except FileExistsError:
        raise FileExistsError(
            f"Folder '{folder_path}' already exists. Cannot create a new one."
        )
