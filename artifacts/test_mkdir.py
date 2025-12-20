import pytest
from mkdir import mkdir
import os
import tempfile
from pathlib import Path


def test_mkdir_success(tmp_path):
    folder_name = tmp_path / "test_folder"
    mkdir(folder_name)
    assert folder_name.exists(), "Directory was not created or does not exist"


def test_mkdir_existing_path(tmp_path):
    folder_name = tmp_path / "test_folder"
    folder_name.mkdir()
    with pytest.raises(
        FileExistsError,
        match=r"\[Errno 17\] File exists: .*",
    ):
        mkdir(folder_name)


def test_mkdir_permission_denied(tmp_path):
    read_only_dir = tmp_path / "read_only"
    read_only_dir.mkdir()
    os.chmod(read_only_dir, 0o444)
    
    folder_name = read_only_dir / "test"
    with pytest.raises(PermissionError):
        mkdir(folder_name)


def test_mkdir_invalid_input_type():
    with pytest.raises(TypeError, match=r"expected str, bytes or os.PathLike object, not NoneType"):
        mkdir(None)

    with pytest.raises(TypeError, match=r"expected str, bytes or os.PathLike object, not int"):
        mkdir(123)
