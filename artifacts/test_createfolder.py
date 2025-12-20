from createfolder import CreateFolder
import pytest
import os
import tempfile
import shutil

def test_create_folder_happy_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        folder_name = "test_folder"
        
        result = CreateFolder().CreateFolder(folder_name)

        assert os.path.exists(folder_name) == True
        assert result == True

def test_create_folder_existing_folder():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        folder_name = "test_folder"
        os.makedirs(folder_name)

        with pytest.raises(ValueError) as excinfo:
            CreateFolder().CreateFolder(folder_name)

        assert str(excinfo.value) == "folder_exists"

def test_create_folder_invalid_name():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        folder_name = "invalid\0name"

        with pytest.raises(RuntimeError):
            CreateFolder().CreateFolder(folder_name)

def test_create_folder_none_input():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        folder_name = None

        with pytest.raises(ValueError) as excinfo:
            CreateFolder().CreateFolder(folder_name)

        assert str(excinfo.value) == "invalid_name"

def test_create_folder_empty_string_input():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        folder_name = ""

        with pytest.raises(ValueError) as excinfo:
            CreateFolder().CreateFolder(folder_name)

        assert str(excinfo.value) == "invalid_name"
