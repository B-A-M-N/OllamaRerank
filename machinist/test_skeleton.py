from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class TestSkeleton:
    __test__ = False
    module_name: str
    func_name: str
    code: str


def build_pytest_skeleton(
    *, 
    module_name: str,
    func_name: str,
    failure_exceptions: Optional[List[str]] = None,
    needs_tmp_path: bool = True,
    intent: str | None = None,
) -> TestSkeleton:
    """
    Returns a pytest file skeleton with non-negotiable structure fixed.
    Is "intent-aware" to generate more specific skeletons for common operations.
    """
    failure_exceptions = sorted(set(e for e in (failure_exceptions or []) if isinstance(e, str) and e.strip()))
    tmp_path_arg = "tmp_path" if needs_tmp_path else ""

    # --- HAPPY PATH SKELETON ---
    happy_path_code = ""
    if intent == "write":
        happy_path_code = f"""def test_{func_name}_success({tmp_path_arg}):
    '''Happy path for a file-writing tool.'''
    # Arrange: Create a file to be modified.
    target_file = {tmp_path_arg} / "test_file.txt"
    target_file.write_text("initial content")

    # Act: Call the function to write new content.
    new_content = "this is the new content"
    {func_name}(str(target_file), new_content)

    # Assert: Verify the file content was overwritten.
    assert target_file.read_text() == new_content
"""
    else:
        # Generic skeleton for copy/move/other file ops
        happy_path_code = f"""def test_{func_name}_success({tmp_path_arg}):
    '''Happy path.'''
    # TODO: Arrange: create any needed files under tmp_path
    # Example for a copy/move tool:
    #   src = {tmp_path_arg} / "source.txt"
    #   dst = {tmp_path_arg} / "dest.txt"
    #   src.write_text("hello")
    #
    # TODO: Act: call the function under test with absolute paths (str(...))
    #   {func_name}(str(src), str(dst))
    #
    # TODO: Assert: verify expected file system state
    #   assert dst.exists()
    #   assert not src.exists() # if it's a move tool
    pass
"""

    # --- FAILURE TESTS SKELETON ---
    failure_tests: List[str] = []
    for exc in failure_exceptions:
        safe_exc = exc.replace(".", "_").replace(" ", "_")
        failure_tests.append(
            f"""def test_{func_name}_raises_{safe_exc}({tmp_path_arg}):
    '''Failure mode: {exc}.'''
    # TODO: Arrange inputs to trigger {exc} exactly as the spec describes.
    with pytest.raises({exc}):
        # TODO: Call the function under test
        {func_name}(...)
"""
        )

    # --- ASSEMBLY ---
    header = f"""import pytest
from {module_name} import {func_name}
"""
    
    main_section = f"""# =========================
# NON-NEGOTIABLE RULES
# =========================
# 1) Do not add or redefine fixtures (especially tmp_path).
# 2) Do not change imports.
# 3) If paths/files are used, ALWAYS build them from tmp_path and pass str(path).
# 4) Only fill in the TODOs below. Do not rewrite the harness.


{happy_path_code}
"""

    code = header.strip() + "\n\n\n" + main_section.strip()

    if failure_tests:
        code += "\n\n\n" + "\n\n".join(failure_tests)

    code += "\n"

    return TestSkeleton(module_name=module_name, func_name=func_name, code=code)


def reject_forbidden_test_patterns(code: str) -> list[str]:
    """
    Fast structural checks. If any hit, we fail+regen instead of 'autofixing' logic.
    """
    errors: list[str] = []

    forbidden_substrings = [
        "def tmp_path(",           # redefining built-in fixture
        " @pytest.fixture\ndef tmp_path",
        "pytest.helpers",          # common hallucination
        "unittest.TestCase",       # style drift
    ]

    for s in forbidden_substrings:
        if s in code:
            errors.append(f"Forbidden test pattern found: {s!r}")

    return errors
