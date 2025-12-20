import pytest
from find_file import find_file
from pathlib import Path


def test_find_file_success(tmp_path: Path):
    """Happy path."""
    file_path = tmp_path / "source.txt"
    file_path.write_text("hello")

    result = find_file(str(file_path))

    assert result == "hello"


def test_find_file_raises_FileNotFoundError():
    """Failure mode: FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        find_file("/nonexistent/path")


def test_find_file_raises_NotADirectoryError(tmp_path: Path):
    """Failure mode: NotADirectoryError."""
    with pytest.raises(NotADirectoryError):
        find_file(str(tmp_path))