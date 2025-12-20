from folder_creator import folder_creator
import pytest
from tempfile import TemporaryDirectory
import os


def test_create_folder_happy_path():
    """Test creating a folder in a temporary directory."""
    with TemporaryDirectory() as temp_dir:
        folder_path = os.path.join(temp_dir, "example_folder")

        folder_creator(folder_path)
        assert os.path.exists(folder_path)


def test_create_folder_already_exists():
    """Test creating a folder that already exists."""
    with TemporaryDirectory() as temp_dir:
        folder_path = os.path.join(temp_dir, "example_folder")
        os.makedirs(folder_path)  # Create the folder first
        
        # The current implementation of folder_creator catches the FileExistsError and does nothing.
        # A better implementation would raise the error or have a flag to ignore it.
        # For now, we will test the current behavior, which is no error is raised.
        try:
            folder_creator(folder_path)
        except FileExistsError:
            pytest.fail("folder_creator raised FileExistsError unexpectedly.")


def test_create_folder_invalid_input():
    """Test creating a folder with an invalid input type."""
    with pytest.raises(TypeError):
        folder_creator(None)
