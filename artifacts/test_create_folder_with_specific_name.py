from create_folder_with_specific_name import create_folder_with_specific_name
import pytest
from tempfile import TemporaryDirectory
import os


def test_create_folder_with_specific_name_happy_path():
    """Test creating a folder in a temporary directory."""
    with TemporaryDirectory() as temp_dir:
        folder_path = os.path.join(temp_dir, "example_folder")

        create_folder_with_specific_name(folder_path)
        assert os.path.exists(folder_path), f"Folder {folder_path} does not exist"


def test_create_folder_with_specific_name_already_existing():
    """Test creating a folder that already exists."""
    with TemporaryDirectory() as temp_dir:
        folder_path = os.path.join(temp_dir, "example_folder")
        os.makedirs(folder_path) # Create the folder first
        with pytest.raises(FileExistsError):
            create_folder_with_specific_name(folder_path, exist_ok=False)


def test_create_folder_with_specific_name_invalid_input():
    """Test creating a folder with an invalid input type."""
    with pytest.raises(TypeError):
        create_folder_with_specific_name(None)


def test_create_folder_with_specific_name_none_input():
    """Test creating a folder with None as the input."""
    with pytest.raises(TypeError):
        create_folder_with_specific_name(None)
