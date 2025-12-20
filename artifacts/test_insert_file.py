import pytest
from insert_file import insert_file
from pathlib import Path


def test_insert_file_success(tmp_path: Path):
    """Happy path."""
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("hello")

    insert_file(str(src), str(dst))

    assert dst.exists()
    assert not src.exists()


def test_insert_file_overwrites_existing_file(tmp_path: Path):
    """Test that insert_file overwrites an existing file."""
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("new content")
    dst.write_text("old content")

    insert_file(str(src), str(dst))

    assert dst.read_text() == "new content"
    assert not src.exists()


def test_insert_file_raises_FileNotFoundError(tmp_path: Path):
    """Failure mode: FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        insert_file(
            str(tmp_path / "non_existent_source.txt"), str(tmp_path / "dest.txt")
        )