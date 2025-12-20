import pytest
from rename_file import rename_file


def test_rename_file_success(tmp_path):
    """Happy path."""
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"

    # Create the source file
    src.write_text("hello")

    # Rename the file
    rename_file(str(src), str(dst))

    # Assert the destination exists and the source does not
    assert dst.exists()
    assert not src.exists()


def test_rename_file_raises_FileExistsError(tmp_path):
    """Failure mode: FileExistsError."""
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"

    # Create both source and destination files
    src.write_text("hello")
    dst.write_text("world")

    with pytest.raises(FileExistsError):
        rename_file(str(src), str(dst))


def test_rename_file_raises_FileNotFoundError(tmp_path):
    """Failure mode: FileNotFoundError."""
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"

    # Try to rename a non-existent source file
    with pytest.raises(FileNotFoundError):
        rename_file(str(src), str(dst))
