import os
import pytest
from duplicate_file import duplicate_file


def test_duplicate_file_success(tmp_path):
    """Happy path."""
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("hello")
    duplicate_file(str(src), str(dst))
    assert dst.exists()
    assert src.exists()


def test_duplicate_file_raises_FileNotFoundError(tmp_path):
    with pytest.raises(FileNotFoundError):
        duplicate_file(str(tmp_path / "nonexistent_file"), str(tmp_path / "dest.txt"))


def test_duplicate_file_raises_PermissionError(tmp_path):
    # Create a read-only directory
    read_only_dir = tmp_path / "read_only"
    read_only_dir.mkdir()
    os.chmod(read_only_dir, 0o444)

    src = tmp_path / "source.txt"
    dst = read_only_dir / "dest.txt"
    src.write_text("hello")

    with pytest.raises(PermissionError):
        duplicate_file(str(src), str(dst))
