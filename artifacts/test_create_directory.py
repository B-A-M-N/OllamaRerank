from create_directory import create_directory
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory


def test_create_directory_happy_path(tmp_path: Path):
    path = tmp_path / "new_dir"
    result = create_directory(str(path))
    assert result, "The directory creation failed."
    assert path.exists(), f"The directory {path} was not created."


def test_create_existing_directory_negative_path(tmp_path: Path):
    path = tmp_path / "existing_dir"
    path.mkdir()
    # The create_directory function should not raise an error if the directory already exists
    # because it uses os.makedirs with exist_ok=True.
    # To test the negative path, we should check what happens when exist_ok=False.
    # However, the current implementation doesn't support passing exist_ok.
    # Let's assume the function is supposed to not fail if the directory exists.
    result = create_directory(str(path))
    assert result, "The directory creation should not fail if the directory already exists."


def test_invalid_input_type_negative_path():
    with pytest.raises(TypeError):
        create_directory(42)


def test_none_input_type_negative_path():
    with pytest.raises(TypeError):
        create_directory(None)
