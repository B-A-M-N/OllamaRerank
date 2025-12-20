import pytest
from insert_newline import insert_newline


def test_insert_newline_no_newline():
    """Tests adding a newline to a string that doesn't have one."""
    assert insert_newline("hello") == "hello\n"


def test_insert_newline_with_newline():
    """Tests a string that already has a newline."""
    assert insert_newline("hello\n") == "hello\n"


def test_insert_newline_empty_string():
    """Tests an empty string."""
    assert insert_newline("") == "\n"


def test_insert_newline_type_error():
    """Tests that a TypeError is raised for non-string input."""
    with pytest.raises(TypeError):
        insert_newline(123)
