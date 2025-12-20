import pytest
import tempfile
import os
from make_directory import make_directory


def test_happy_path():
    with tempfile.TemporaryDirectory() as tmpdirname:
        dir_path = os.path.join(tmpdirname, "mydir")
        make_directory(dir_path)
        assert os.path.isdir(dir_path)


def test_already_exists():
    with tempfile.TemporaryDirectory() as tmpdirname:
        dir_path = os.path.join(tmpdirname, "mydir")
        make_directory(dir_path)
        assert os.path.isdir(dir_path)
        with pytest.raises(FileExistsError):
            make_directory(dir_path)


def test_invalid_input():
    with pytest.raises(TypeError):
        make_directory(123)
    with pytest.raises(TypeError):
        make_directory(None)

def test_invalid_input_nonexistent_parent():
    with pytest.raises(OSError):
        make_directory("/path/to/nonexistent/directory")
