import os
import fnmatch


def search_files(root_dir: str, pattern: str, *, recursive: bool = True) -> list[str]:
    """Searches for files matching a pattern in a directory."""
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"The root directory '{root_dir}' does not exist.")
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"The root path '{root_dir}' is not a directory.")

    found_paths: list[str] = []

    if recursive:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if fnmatch.fnmatch(filename, pattern):
                    found_paths.append(os.path.join(dirpath, filename))
    else:
        # Non-recursive: only immediate files in root_dir
        for name in os.listdir(root_dir):
            path = os.path.join(root_dir, name)
            if os.path.isfile(path) and fnmatch.fnmatch(name, pattern):
                found_paths.append(path)

    return found_paths
