import pytest
from pathlib import Path
import os
from search_files import search_files


def test_search_files_success(tmp_path: Path):
    """Tests that files are found correctly."""
    (tmp_path / "a.txt").touch()
    (tmp_path / "b.txt").touch()
    (tmp_path / "c.log").touch()

    result = search_files(str(tmp_path), "*.txt")

    assert len(result) == 2
    assert str(tmp_path / "a.txt") in result
    assert str(tmp_path / "b.txt") in result


def test_search_files_recursive(tmp_path: Path):
    """Tests recursive search."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (tmp_path / "a.txt").touch()
    (sub_dir / "b.txt").touch()

    result = search_files(str(tmp_path), "*.txt", recursive=True)

    assert len(result) == 2
    assert str(tmp_path / "a.txt") in result
    assert str(sub_dir / "b.txt") in result


def test_search_files_not_recursive(tmp_path: Path):
    """Tests non-recursive search."""
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (tmp_path / "a.txt").touch()
    (sub_dir / "b.txt").touch()

    result = search_files(str(tmp_path), "*.txt", recursive=False)

    assert len(result) == 1
    assert str(tmp_path / "a.txt") in result


def test_search_files_root_not_found(tmp_path: Path):
    """Tests that FileNotFoundError is raised if the root directory does not exist."""
    with pytest.raises(FileNotFoundError):
        search_files("non_existent_dir", "*.txt")


def test_search_files_root_is_not_a_directory(tmp_path: Path):
    """Tests that NotADirectoryError is raised if the root path is not a directory."""
    file_path = tmp_path / "file.txt"
    file_path.touch()
    with pytest.raises(NotADirectoryError):
        search_files(str(file_path), "*.txt")
