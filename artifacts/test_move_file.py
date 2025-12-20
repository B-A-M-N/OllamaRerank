import pytest
from pathlib import Path
import os
from move_file import move_file


def test_move_file_success(tmp_path: Path):
    """Tests that the file is moved successfully."""
    src_file = tmp_path / "source.txt"
    dst_file = tmp_path / "destination.txt"
    src_file.write_text("hello world")

    move_file(str(src_file), str(dst_file))

    assert dst_file.exists()
    assert not src_file.exists()  # For move, source should NOT exist
    assert dst_file.read_text() == "hello world"


def test_move_file_source_not_found(tmp_path: Path):
    """Tests that FileNotFoundError is raised if the source file does not exist."""
    dst_file = tmp_path / "destination.txt"

    with pytest.raises(FileNotFoundError):
        move_file("non_existent_file.txt", str(dst_file))


def test_move_file_destination_exists(tmp_path: Path):
    """Tests that FileExistsError is raised if the destination file already exists."""
    src_file = tmp_path / "source.txt"
    dst_file = tmp_path / "destination.txt"
    src_file.write_text("hello world")
    dst_file.touch()

    with pytest.raises(FileExistsError):
        move_file(str(src_file), str(dst_file))
