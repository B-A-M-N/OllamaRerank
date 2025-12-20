import os
import shutil
import pathlib

def sort_files(input_dir: str) -> None:
    """
    Sorts all files in a directory into subdirectories based on their file extension.

    Args:
        input_dir (str): The path to the directory containing the files to be sorted.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        ValueError: If the input_dir is not a directory.
        IOError: If there's an error during file operations (e.g., permission denied).
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a directory: {input_dir}")

    # Ensure input_dir is a Path object for easier manipulation
    input_path = pathlib.Path(input_dir)

    for item in input_path.iterdir():
        if item.is_file():
            # Get the extension (e.g., ".txt", ".jpg")
            # If no extension, `suffix` is empty
            extension = item.suffix

            # Use the extension as the subdirectory name, without the leading dot
            # If no extension, use a default name like "no_extension"
            subdir_name = extension[1:] if extension else "no_extension"
            
            destination_subdir = input_path / subdir_name
            
            # Create the destination subdirectory if it doesn't exist
            destination_subdir.mkdir(parents=True, exist_ok=True)

            try:
                # Construct the full destination path for the file
                destination_file_path = destination_subdir / item.name
                shutil.move(str(item), str(destination_file_path))
            except shutil.Error as e:
                raise IOError(f"Error moving file {item.name} to {destination_file_path}: {e}")
