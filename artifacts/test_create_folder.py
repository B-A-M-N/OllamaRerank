import pytest
from pathlib import Path
from create_folder import create_folder


def test_happy_path(tmp_path: Path):
    test_folder = tmp_path / "my_folder"
    create_folder(str(test_folder))
    assert test_folder.is_dir()


def test_folder_already_exists(tmp_path: Path):
    test_folder = tmp_path / "my_folder"
    test_folder.mkdir()
    with pytest.raises(ValueError):
        create_folder(str(test_folder))
