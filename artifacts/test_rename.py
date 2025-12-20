import os
from pathlib import Path
import pytest
from rename import rename

def test_rename_success(tmp_path: Path):
    os.chdir(tmp_path)
    old_name = "source.txt"
    new_name = "destination.txt"
    
    (tmp_path / old_name).write_text("Hello world!")
    
    rename(old_name, new_name)
    
    assert not (tmp_path / old_name).exists()
    assert (tmp_path / new_name).exists()


def test_rename_raises_FileNotFoundError(tmp_path: Path):
    os.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        rename("non-existent.txt", "destination.txt")


def test_rename_raises_ValueError_if_new_name_exists(tmp_path: Path):
    os.chdir(tmp_path)
    old_name = "source.txt"
    new_name = "destination.txt"
    
    (tmp_path / old_name).write_text("Hello world!")
    (tmp_path / new_name).write_text("existing file")
    
    with pytest.raises(ValueError):
        rename(old_name, new_name)