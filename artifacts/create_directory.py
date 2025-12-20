import os


def create_directory(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except PermissionError as e:
        raise PermissionError(f"Permission denied to create directory at {path}: {e}")
    except OSError as e:
        raise OSError(f"OS error occurred while creating directory at {path}: {e}")
    except IsADirectoryError as e:
        raise IsADirectoryError(f"{path} is already a directory")
