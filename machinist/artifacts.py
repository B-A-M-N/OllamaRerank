from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Tuple
from datetime import datetime

from .config import MachinistConfig
from .llm_interface import LLMInterface
from .parsing import normalize_snippet, to_safe_module_name
from .registry import ToolSpec


class Artifacts:
    def __init__(self, llm_interface: LLMInterface, config: MachinistConfig):
        self.llm_interface = llm_interface
        self.config = config

    def write_artifacts(self, spec: ToolSpec, code: str, tests: str) -> Tuple[Path, Path]:
        tools_dir = Path.cwd() / "artifacts"
        tools_dir.mkdir(parents=True, exist_ok=True)
        safe_name = to_safe_module_name(spec.name)
        source_path = tools_dir / f"{safe_name}.py"
        tests_path = tools_dir / f"test_{safe_name}.py"
        attempts = 0
        max_attempts = 3
        current_code = code
        current_tests = tests
        while True:
            attempts += 1
            # Quick preflight: if the LLM didn't return any test_* functions, regenerate with strict prompt.
            if "def test_" not in current_tests:
                # Persist what we got so far to help debug why the model isn't emitting tests.
                debug_dir = Path.cwd() / "artifacts"
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / f"debug_{to_safe_module_name(spec.name)}_tests_raw.txt").write_text(
                    current_tests, encoding="utf-8"
                )
                if attempts >= max_attempts:
                    raise ValueError(
                        "No test functions detected after multiple attempts. "
                        f"See artifacts/debug_{to_safe_module_name(spec.name)}_tests_raw.txt for the last LLM output."
                    )
                current_tests = self.llm_interface.generate_tests(spec, stream=False, strict=True, safe_name=to_safe_module_name(spec.name))
                continue
            try:
                cleaned_code, cleaned_tests = self._prepare_artifacts(spec, current_code, current_tests)
                source_path.write_text(cleaned_code, encoding="utf-8")
                tests_path.write_text(cleaned_tests, encoding="utf-8")
                # Persist raw generated tests for debugging if they end up empty after sanitization.
                if not cleaned_tests.strip():
                    (tools_dir / f"debug_{safe_name}_tests_raw.txt").write_text(current_tests, encoding="utf-8")
                return source_path, tests_path
            except ValueError as exc:
                if attempts >= max_attempts:
                    raise
                
                if "Generated code is empty" in str(exc):
                    print("Generated code was empty, regenerating...")
                    current_code = self.llm_interface.generate_implementation(spec, stream=False)
                    continue

                # Try regenerating tests with a stricter prompt.
                current_tests = self.llm_interface.generate_tests(spec, stream=False, strict=True, safe_name=to_safe_module_name(spec.name))

    def _prepare_artifacts(self, spec: ToolSpec, code: str, tests: str) -> Tuple[str, str]:
        """
        Normalize raw code and tests emitted by the LLM.
        """
        cleaned_code = normalize_snippet(code)
        cleaned_tests = normalize_snippet(tests)

        # Remove the incorrect import statement
        cleaned_tests = cleaned_tests.replace("from pytest import tmp_path", "")

        if not cleaned_code:
            raise ValueError("Generated code is empty after normalization.")

        allow_empty_tests = getattr(self.config, "allow_empty_tests", False)
        if not cleaned_tests and not allow_empty_tests:
            raise ValueError("No test functions detected after normalization.")

        module_name = to_safe_module_name(spec.name)
        cleaned_tests = self._ensure_imports(cleaned_tests, module_name, cleaned_code, spec.name)
        return cleaned_code, cleaned_tests
    
    def _ensure_imports(self, test_code: str, module_name: str, tool_code: str, func_name: str) -> str:
        
        # Always import the function under test
        imports_to_add = {f"from {module_name} import {func_name}"}

        # Statically analyze the test code for undefined names
        try:
            tree = ast.parse(test_code)
            defined_names = set()
            
            # Find all imported names
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name)
            
            # Find all function and variable names
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    defined_names.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    defined_names.add(node.id)

            # Find all used names that are not defined
            undefined_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id not in defined_names:
                        undefined_names.add(node.id)

            # Map common names to modules
            common_imports = {
                "pytest": "import pytest",
                "Path": "from pathlib import Path",
                "TemporaryDirectory": "from tempfile import TemporaryDirectory",
                "rmtree": "from shutil import rmtree",
            }
            
            for name in undefined_names:
                if name in common_imports:
                    imports_to_add.add(common_imports[name])

        except SyntaxError:
            # If the code is not valid python, we can't parse it.
            # Fallback to the previous logic.
            tool_imports = re.findall(r"^(?:import|from)\s+.*", tool_code, re.MULTILINE)
            test_imports = re.findall(r"^(?:import|from)\s+.*", test_code, re.MULTILINE)
            
            imports_to_add.update(tool_imports)
            
            if "import pytest" not in test_imports:
                imports_to_add.add("import pytest")
        
        # Remove imports that are already in the test code
        test_imports = re.findall(r"^(?:import|from)\s+.*", test_code, re.MULTILINE)
        for imp in test_imports:
            if imp in imports_to_add:
                imports_to_add.remove(imp)

        return "\n".join(sorted(list(imports_to_add))) + "\n\n" + test_code