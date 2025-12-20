import os


def folder_creator(folder_name: str) -> None:
    """
    Creates a new folder with the specified name.

    Args:
        folder_name (str): The name of the folder to be created.

    Returns:
        None
    """
    try:
        os.mkdir(folder_name)
    except FileExistsError as e:
        # Handle the exception here
        pass
