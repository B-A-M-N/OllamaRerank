from __future__ import annotations

from .registry import ToolSpec

def get_copy_file_spec() -> ToolSpec:
    """
    Returns a hardcoded, correct ToolSpec for a 'copy_file' operation.
    """
    return ToolSpec(
        name="copy_file",
        signature="def copy_file(src_path: str, dst_path: str) -> None:",
        docstring="Copies a file from a source path to a destination path.",
        imports=["os", "shutil"],
        inputs={
            "src_path": "The path to the source file.",
            "dst_path": "The path to the destination file.",
        },
        outputs={},
        failure_modes=[
            {"exception": "FileNotFoundError", "reason": "The source file does not exist."},
            {"exception": "FileExistsError", "reason": "The destination file already exists."},
        ],
        deterministic=True,
    )

def get_copy_file_tests(spec: ToolSpec, safe_module_name: str) -> str:
    """
    Returns a hardcoded, correct pytest file for the 'copy_file' operation.
    """
    test_content = """
import pytest
from pathlib import Path
import os
from {safe_module_name} import {func_name}

def test_copy_file_success(tmp_path: Path):
    \"\"\"Tests that the file is copied successfully.\"\"\"
    src_file = tmp_path / "source.txt"
    dst_file = tmp_path / "destination.txt"
    src_file.write_text("hello world")

    {func_name}(str(src_file), str(dst_file))

    assert dst_file.exists()
    assert src_file.exists() # For copy, source should still exist
    assert dst_file.read_text() == "hello world"

def test_copy_file_source_not_found(tmp_path: Path):
    \"\"\"Tests that FileNotFoundError is raised if the source file does not exist.\"\"\"
    dst_file = tmp_path / "destination.txt"

    with pytest.raises(FileNotFoundError):
        {func_name}("non_existent_file.txt", str(dst_file))

def test_copy_file_destination_exists(tmp_path: Path):
    \"\"\"Tests that FileExistsError is raised if the destination file already exists.\"\"\"
    src_file = tmp_path / "source.txt"
    dst_file = tmp_path / "destination.txt"
    src_file.write_text("hello world")
    dst_file.touch()

    with pytest.raises(FileExistsError):
        {func_name}(str(src_file), str(dst_file))
"""
    return test_content.format(safe_module_name=safe_module_name, func_name=spec.name)


def get_move_file_spec() -> ToolSpec:
    """
    Returns a hardcoded, correct ToolSpec for a 'move_file' operation.
    """
    return ToolSpec(
        name="move_file",
        signature="def move_file(src_path: str, dst_path: str) -> None:",
        docstring="Moves a file from a source path to a destination path.",
        imports=["os", "shutil"],
        inputs={
            "src_path": "The path to the source file.",
            "dst_path": "The path to the destination file.",
        },
        outputs={},
        failure_modes=[
            {"exception": "FileNotFoundError", "reason": "The source file does not exist."},
            {"exception": "FileExistsError", "reason": "The destination file already exists."},
        ],
        deterministic=True,
    )


def get_move_file_tests(spec: ToolSpec, safe_module_name: str) -> str:
    """
    Returns a hardcoded, correct pytest file for the 'move_file' operation.
    """
    test_content = """
import pytest
from pathlib import Path
import os
from {safe_module_name} import {func_name}

def test_move_file_success(tmp_path: Path):
    \"\"\"Tests that the file is moved successfully.\"\"\"
    src_file = tmp_path / "source.txt"
    dst_file = tmp_path / "destination.txt"
    src_file.write_text("hello world")

    {func_name}(str(src_file), str(dst_file))

    assert dst_file.exists()
    assert not src_file.exists() # For move, source should NOT exist
    assert dst_file.read_text() == "hello world"

def test_move_file_source_not_found(tmp_path: Path):
    \"\"\"Tests that FileNotFoundError is raised if the source file does not exist.\"\"\"
    dst_file = tmp_path / "destination.txt"

    with pytest.raises(FileNotFoundError):
        {func_name}("non_existent_file.txt", str(dst_file))

def test_move_file_destination_exists(tmp_path: Path):
    \"\"\"Tests that FileExistsError is raised if the destination file already exists.\"\"\"
    src_file = tmp_path / "source.txt"
    dst_file = tmp_path / "destination.txt"
    src_file.write_text("hello world")
    dst_file.touch()

    with pytest.raises(FileExistsError):
        {func_name}(str(src_file), str(dst_file))
"""
    return test_content.format(safe_module_name=safe_module_name, func_name=spec.name)


def get_delete_file_spec() -> ToolSpec:
    """
    Returns a hardcoded, correct ToolSpec for a 'delete_file' operation.
    """
    return ToolSpec(
        name="delete_file",
        signature="def delete_file(path: str) -> None:",
        docstring="Deletes a file.",
        imports=["os"],
        inputs={"path": "The path to the file to delete."},
        outputs={},
        failure_modes=[
            {"exception": "FileNotFoundError", "reason": "The file does not exist."},
        ],
        deterministic=True,
    )


def get_delete_file_tests(spec: ToolSpec, safe_module_name: str) -> str:
    """
    Returns a hardcoded, correct pytest file for the 'delete_file' operation.
    """
    test_content = """
import pytest
from pathlib import Path
import os
from {safe_module_name} import {func_name}

def test_delete_file_success(tmp_path: Path):
    \"\"\"Tests that the file is deleted successfully.\"\"\"
    file_to_delete = tmp_path / "file.txt"
    file_to_delete.write_text("hello world")

    {func_name}(str(file_to_delete))

    assert not file_to_delete.exists()

def test_delete_file_not_found(tmp_path: Path):
    \"\"\"Tests that FileNotFoundError is raised if the file does not exist.\"\"\"
    with pytest.raises(FileNotFoundError):
        {func_name}("non_existent_file.txt")
"""
    return test_content.format(safe_module_name=safe_module_name, func_name=spec.name)


def get_read_file_spec() -> ToolSpec:
    """
    Returns a hardcoded, correct ToolSpec for a 'read_file' operation.
    """
    return ToolSpec(
        name="read_file",
        signature="def read_file(path: str) -> str:",
        docstring="Reads the contents of a file and returns it as a string.",
        imports=["os"],
        inputs={"path": "The path to the file to read."},
        outputs={"contents": "The contents of the file as a string."},
        failure_modes=[
            {"exception": "FileNotFoundError", "reason": "The file does not exist."},
            {"exception": "IsADirectoryError", "reason": "The path points to a directory, not a file."},
        ],
        deterministic=True,
    )


def get_read_file_tests(spec: ToolSpec, safe_module_name: str) -> str:
    """
    Returns a hardcoded, correct pytest file for the 'read_file' operation.
    """
    test_content = """
import pytest
from pathlib import Path
import os
from {safe_module_name} import {func_name}

def test_read_file_success(tmp_path: Path):
    \"\"\"Tests that the file is read successfully.\"\"\"
    file_to_read = tmp_path / "file.txt"
    content = "hello world"
    file_to_read.write_text(content)

    result = {func_name}(str(file_to_read))

    assert result == content

def test_read_file_not_found(tmp_path: Path):
    \"\"\"Tests that FileNotFoundError is raised if the file does not exist.\"\"\"
    with pytest.raises(FileNotFoundError):
        {func_name}("non_existent_file.txt")

def test_read_file_is_a_directory(tmp_path: Path):
    \"\"\"Tests that IsADirectoryError is raised if the path is a directory.\"\"\"
    with pytest.raises(IsADirectoryError):
        {func_name}(str(tmp_path))
"""
    return test_content.format(safe_module_name=safe_module_name, func_name=spec.name)


def get_search_files_spec() -> ToolSpec:
    """
    Returns a hardcoded, correct ToolSpec for a 'search_files' operation.
    """
    return ToolSpec(
        name="search_files",
        signature="def search_files(root_dir: str, pattern: str, *, recursive: bool = True) -> list[str]:",
        docstring="Searches for files matching a pattern in a directory.",
        imports=["os", "fnmatch"],
        inputs={
            "root_dir": "The directory to start the search from.",
            "pattern": "The glob pattern to match filenames against (e.g., '*.txt').",
            "recursive": "If True, searches directories recursively.",
        },
        outputs={"found_paths": "A list of absolute paths to the found files."},
        failure_modes=[
            {"exception": "FileNotFoundError", "reason": "The root directory does not exist."},
            {"exception": "NotADirectoryError", "reason": "The root path is not a directory."},
        ],
        deterministic=True,
    )


def get_search_files_tests(spec: ToolSpec, safe_module_name: str) -> str:
    """
    Returns a hardcoded, correct pytest file for the 'search_files' operation.
    """
    test_content = """
import pytest
from pathlib import Path
import os
from {safe_module_name} import {func_name}

def test_search_files_success(tmp_path: Path):
    \"\"\"Tests that files are found correctly.\"\"\"
    (tmp_path / "a.txt").touch()
    (tmp_path / "b.txt").touch()
    (tmp_path / "c.log").touch()

    result = {func_name}(str(tmp_path), "*.txt")

    assert len(result) == 2
    assert str(tmp_path / "a.txt") in result
    assert str(tmp_path / "b.txt") in result

def test_search_files_recursive(tmp_path: Path):
    \"\"\"Tests recursive search.\"\"\"
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (tmp_path / "a.txt").touch()
    (sub_dir / "b.txt").touch()

    result = {func_name}(str(tmp_path), "*.txt", recursive=True)

    assert len(result) == 2
    assert str(tmp_path / "a.txt") in result
    assert str(sub_dir / "b.txt") in result

def test_search_files_not_recursive(tmp_path: Path):
    \"\"\"Tests non-recursive search.\"\"\"
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (tmp_path / "a.txt").touch()
    (sub_dir / "b.txt").touch()

    result = {func_name}(str(tmp_path), "*.txt", recursive=False)

    assert len(result) == 1
    assert str(tmp_path / "a.txt") in result

def test_search_files_root_not_found(tmp_path: Path):
    \"\"\"Tests that FileNotFoundError is raised if the root directory does not exist.\"\"\"
    with pytest.raises(FileNotFoundError):
        {func_name}("non_existent_dir", "*.txt")

def test_search_files_root_is_not_a_directory(tmp_path: Path):
    \"\"\"Tests that NotADirectoryError is raised if the root path is not a directory.\"\"\"
    file_path = tmp_path / "file.txt"
    file_path.touch()
    with pytest.raises(NotADirectoryError):
        {func_name}(str(file_path), "*.txt")
"""
    return test_content.format(safe_module_name=safe_module_name, func_name=spec.name)


def get_search_files_impl(spec: ToolSpec) -> str:
    """
    Returns a hardcoded, correct implementation for the 'search_files' operation.
    """
    return '''
import os
import fnmatch

def search_files(root_dir: str, pattern: str, *, recursive: bool = True) -> list[str]:
    """Searches for files matching a pattern in a directory."""
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"The root directory '{root_dir}' does not exist.")
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"The root path '{root_dir}' is not a directory.")

    found_paths: list[str] = []

    if recursive:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if fnmatch.fnmatch(filename, pattern):
                    found_paths.append(os.path.join(dirpath, filename))
    else:
        # Non-recursive: only immediate files in root_dir
        for name in os.listdir(root_dir):
            path = os.path.join(root_dir, name)
            if os.path.isfile(path) and fnmatch.fnmatch(name, pattern):
                found_paths.append(path)

    return found_paths
'''


def get_insert_newline_spec() -> ToolSpec:
    """
    Returns a hardcoded, correct ToolSpec for an 'insert_newline' operation.
    """
    return ToolSpec(
        name="insert_newline",
        signature="def insert_newline(text: str) -> str:",
        docstring="Ensures a string ends with a newline character.",
        imports=[],
        inputs={"text": "The input string."},
        outputs={"text": "The string, with a newline at the end."},
        failure_modes=[
            {"exception": "TypeError", "reason": "The input is not a string."},
        ],
        deterministic=True,
    )


def get_insert_newline_tests(spec: ToolSpec, safe_module_name: str) -> str:
    """
    Returns a hardcoded, correct pytest file for the 'insert_newline' operation.
    """
    test_content = """
import pytest
from {safe_module_name} import {func_name}

def test_insert_newline_no_newline():
    \"\"\"Tests adding a newline to a string that doesn't have one.\"\"\"
    assert {func_name}("hello") == "hello\\n"

def test_insert_newline_with_newline():
    \"\"\"Tests a string that already has a newline.\"\"\"
    assert {func_name}("hello\\n") == "hello\\n"

def test_insert_newline_empty_string():
    \"\"\"Tests an empty string.\"\"\"
    assert {func_name}("") == "\\n"

def test_insert_newline_type_error():
    \"\"\"Tests that a TypeError is raised for non-string input.\"\"\"
    with pytest.raises(TypeError):
        {func_name}(123)
"""
    return test_content.format(safe_module_name=safe_module_name, func_name=spec.name)


def get_insert_newline_impl(spec: ToolSpec) -> str:
    """
    Returns a hardcoded, correct implementation for the 'insert_newline' operation.
    """
    return '''
def insert_newline(text: str) -> str:
    """Ensures a string ends with a newline character."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    if not text.endswith("\\n"):
        return text + "\\n"
    return text
'''


def get_document_search_spec() -> ToolSpec:
    """
    Returns a hardcoded, correct ToolSpec for a 'document_search' operation.
    """
    return ToolSpec(
        name="document_search",
        signature="def document_search(path: str, needle: str, *, case_sensitive: bool = True) -> list[tuple[int, str]]:",
        docstring="Finds lines in a file containing a specific string or pattern.",
        imports=["os", "re"],
        inputs={
            "path": "The path to the file to search.",
            "needle": "The string or regex pattern to search for.",
            "case_sensitive": "If True, the search is case-sensitive.",
        },
        outputs={"matches": "A list of tuples, where each tuple contains a line number and the line text."},
        failure_modes=[
            {"exception": "FileNotFoundError", "reason": "The file does not exist."},
            {"exception": "IsADirectoryError", "reason": "The path points to a directory, not a file."},
        ],
        deterministic=True,
    )


def get_document_search_tests(spec: ToolSpec, safe_module_name: str) -> str:
    """
    Returns a hardcoded, correct pytest file for the 'document_search' operation.
    """
    test_content = """
import pytest
from pathlib import Path
import os
from {safe_module_name} import {func_name}

@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello world\\nThis is a test\\nHELLO AGAIN\\n")
    return file_path

def test_document_search_success(sample_file: Path):
    \"\"\"Tests that matching lines are found correctly (case-sensitive).\"\"\"
    matches = {func_name}(str(sample_file), "Hello")
    assert len(matches) == 1
    assert matches[0] == (1, "Hello world")

def test_document_search_case_insensitive(sample_file: Path):
    \"\"\"Tests case-insensitive search.\"\"\"
    matches = {func_name}(str(sample_file), "Hello", case_sensitive=False)
    assert len(matches) == 2
    assert matches[0] == (1, "Hello world")
    assert matches[1] == (3, "HELLO AGAIN")

def test_document_search_no_matches(sample_file: Path):
    \"\"\"Tests that an empty list is returned when no matches are found.\"\"\"
    matches = {func_name}(str(sample_file), "non-existent")
    assert len(matches) == 0

def test_document_search_file_not_found(tmp_path: Path):
    \"\"\"Tests that FileNotFoundError is raised if the file does not exist.\"\"\"
    with pytest.raises(FileNotFoundError):
        {func_name}("non_existent_file.txt", "test")

def test_document_search_is_a_directory(tmp_path: Path):
    \"\"\"Tests that IsADirectoryError is raised if the path is a directory.\"\"\"
    with pytest.raises(IsADirectoryError):
        {func_name}(str(tmp_path), "test")
"""
    return test_content.format(safe_module_name=safe_module_name, func_name=spec.name)


def get_document_search_impl(spec: ToolSpec) -> str:
    """
    Returns a hardcoded, correct implementation for the 'document_search' operation.
    """
    return '''
import os
import re

def document_search(path: str, needle: str, *, case_sensitive: bool = True) -> list[tuple[int, str]]:
    """Finds lines in a file containing a specific string or pattern."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file '{path}' does not exist.")
    if os.path.isdir(path):
        raise IsADirectoryError(f"The path '{path}' is a directory, not a file.")

    matches = []
    flags = 0 if case_sensitive else re.IGNORECASE
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if re.search(needle, line, flags):
                    matches.append((i, line.strip()))
    except UnicodeDecodeError:
        # For simplicity, we'll just return no matches for binary files.
        # A more advanced tool might handle different encodings.
        pass

    return matches
'''
