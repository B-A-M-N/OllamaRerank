import pytest
import os
from copy_file import copy_file


def test_copy_file_success(tmp_path):
    """Happy path."""
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"

    src.write_text("hello")
    copy_file(str(src), str(dst))

    assert dst.exists()
    assert src.exists()


def test_copy_file_raises_FileExistsError(tmp_path):
    """Failure mode: FileExistsError."""
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("hello")
    dst.write_text("world")
    os.chmod(dst, 0o444) # make readonly
    with pytest.raises(FileExistsError):
        copy_file(str(src), str(dst))


def test_copy_file_raises_FileNotFoundError(tmp_path):
    """Failure mode: FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        copy_file(str(tmp_path / "non_existent_src.txt"), str(tmp_path / "dest.txt"))
