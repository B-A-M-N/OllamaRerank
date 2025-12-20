import os
import pytest
from file_renamer import file_renamer
from tempfile import TemporaryDirectory


def test_file_renamer_success():
    with TemporaryDirectory() as temp_dir:
        src = os.path.join(temp_dir, "source.txt")
        dst = "destination.txt"
        
        with open(src, "w") as f:
            f.write("hello")
            
        file_renamer(src, dst)
        
        assert not os.path.exists(src)
        assert os.path.exists(os.path.join(temp_dir, dst))

def test_file_renamer_raises_ValueError_if_file_not_found():
    with pytest.raises(ValueError):
        file_renamer("non_existent_file.txt", "new_name.txt")

def test_file_renamer_raises_PermissionError(tmp_path):
    src = tmp_path / "source.txt"
    src.write_text("hello")
    
    # Create a read-only directory
    read_only_dir = tmp_path / "read_only"
    read_only_dir.mkdir()
    os.chmod(read_only_dir, 0o444)
    
    dst = read_only_dir / "dest.txt"
    
    with pytest.raises(PermissionError):
        file_renamer(str(src), str(dst))

def test_file_renamer_invalid_input():
    with pytest.raises(ValueError):
        file_renamer(123, "new_name.txt")