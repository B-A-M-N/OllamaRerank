import os
import shutil
import pytest
from sort_files import sort_files # Ensure this import is present and correct

def test_sort_files_success(tmp_path):
    """
    Test sorting of various file types into extension-based subdirectories.
    """
    # Create a temporary directory structure for testing
    input_dir = tmp_path / "test_input"
    input_dir.mkdir()

    # Create dummy files
    (input_dir / "file1.txt").write_text("content1")
    (input_dir / "document.pdf").write_text("content2")
    (input_dir / "image.jpg").write_text("content3")
    (input_dir / "archive.tar.gz").write_text("content4") # Multi-part extension
    (input_dir / "no_extension_file").write_text("content5")
    (input_dir / "another.txt").write_text("content6")

    # Run the sort_files function
    sort_files(str(input_dir))

    # Assert that files are moved to correct subdirectories
    assert (input_dir / "txt" / "file1.txt").exists()
    assert (input_dir / "pdf" / "document.pdf").exists()
    assert (input_dir / "jpg" / "image.jpg").exists()
    assert (input_dir / "gz" / "archive.tar.gz").exists() # Should use the last part of multi-part extension
    assert (input_dir / "no_extension" / "no_extension_file").exists()
    assert (input_dir / "txt" / "another.txt").exists()

    # Assert that original files no longer exist in the input_dir
    assert not (input_dir / "file1.txt").exists()
    assert not (input_dir / "document.pdf").exists()
    assert not (input_dir / "image.jpg").exists()
    assert not (input_dir / "archive.tar.gz").exists()
    assert not (input_dir / "no_extension_file").exists()
    assert not (input_dir / "another.txt").exists()

    # Assert that subdirectories are created
    assert (input_dir / "txt").is_dir()
    assert (input_dir / "pdf").is_dir()
    assert (input_dir / "jpg").is_dir()
    assert (input_dir / "gz").is_dir()
    assert (input_dir / "no_extension").is_dir()

def test_sort_files_empty_directory(tmp_path):
    """
    Test sorting on an empty directory.
    """
    input_dir = tmp_path / "empty_dir"
    input_dir.mkdir()

    sort_files(str(input_dir))

    # Assert that the directory remains empty (except for potential .pytest_cache if created)
    assert len(list(input_dir.iterdir())) == 0

def test_sort_files_with_subdirectories(tmp_path):
    """
    Test sorting when the input directory contains subdirectories.
    These should be ignored by the sorting logic.
    """
    input_dir = tmp_path / "parent_dir"
    input_dir.mkdir()
    (input_dir / "file.txt").write_text("test")
    (input_dir / "subdir").mkdir()
    (input_dir / "subdir" / "nested_file.json").write_text("{}")

    sort_files(str(input_dir))

    # Assert that the file is sorted
    assert (input_dir / "txt" / "file.txt").exists()
    assert not (input_dir / "file.txt").exists()

    # Assert that the subdirectory and its contents are untouched
    assert (input_dir / "subdir").is_dir()
    assert (input_dir / "subdir" / "nested_file.json").exists()

def test_sort_files_non_existent_directory():
    """
    Test FileNotFoundError when input directory does not exist.
    """
    with pytest.raises(FileNotFoundError, match="Input directory does not exist"):
        sort_files("non_existent_path")

def test_sort_files_input_is_file(tmp_path):
    """
    Test ValueError when input path is a file, not a directory.
    """
    test_file = tmp_path / "a_file.txt"
    test_file.write_text("some content")

    with pytest.raises(ValueError, match="Input path is not a directory"):
        sort_files(str(test_file))

def test_sort_files_permissions_error_source(tmp_path):
    """
    Test handling of permission errors when moving files (e.g., read-only source file).
    """
    input_dir = tmp_path / "permissions_test_source"
    input_dir.mkdir()
    
    file_to_move = input_dir / "no_read.txt"
    file_to_move.write_text("important data")
    
    # Make the file read-only (simulate permission issue for shutil.move if it tries to delete/rename)
    os.chmod(str(file_to_move), 0o444) # Read-only for owner, group, others

    # shutil.move might still succeed if it only needs read access to copy and then delete
    # or it might fail if it tries to rename directly.
    # The current implementation will move it, but deleting the original might fail later.
    # It's hard to consistently trigger an IOError for shutil.move with read-only source
    # when the destination is writable. Let's try making the destination unwriteable instead.

    # Re-evaluate this test: shutil.move usually requires write permission on the *directory*
    # containing the source file to remove it after moving. Making the file itself read-only
    # often doesn't prevent `shutil.move` from succeeding if the directory is writable.
    # Let's test by making the *input_dir* unwritable, preventing creation of subdirectories.
    os.chmod(str(file_to_move), 0o644) # Reset permissions for file for next test
    shutil.rmtree(str(input_dir)) # Clean up for the next permission test

    input_dir = tmp_path / "permissions_test_dest"
    input_dir.mkdir()
    (input_dir / "can_move.txt").write_text("data")

    # Make the input_dir unwritable, preventing subdir creation and moves
    os.chmod(str(input_dir), 0o555) # Read-only directory

    with pytest.raises(IOError):
        # This will fail because it can't create subdirectories or move into them
        sort_files(str(input_dir))
    
    os.chmod(str(input_dir), 0o755) # Restore permissions for cleanup

