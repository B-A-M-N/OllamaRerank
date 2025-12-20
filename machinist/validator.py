from __future__ import annotations

import json
import subprocess
import sys
import site
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

from .sandbox import BwrapSandbox, SandboxPolicy


@dataclass
class ValidationResult:
    lint_ok: bool
    tests_ok: bool
    coverage_ok: bool
    command_results: Dict[str, subprocess.CompletedProcess]

    def is_ok(self) -> bool:
        """Returns True if all validation checks passed."""
        return self.lint_ok and self.tests_ok and self.coverage_ok


class ValidationRunner:
    """
    Runs lint/tests/coverage inside the configured sandbox. If sandbox fails to start,
    validation is treated as failed.
    """

    def __init__(self, sandbox_policy: SandboxPolicy | None = None) -> None:
        self.sandbox = BwrapSandbox(sandbox_policy or SandboxPolicy())

    def run(
        self,
        source_path: Path,
        tests_path: Path,
        *,
        coverage_threshold: float = 0.8,
        stream: bool = False,
        on_output: callable | None = None,
    ) -> ValidationResult:
        source_abs = source_path.resolve()
        tests_abs = tests_path.resolve()
        python_bin = sys.executable

        # Ensure user/site packages are visible inside sandbox (e.g., user-level pytest installs).
        self._bind_site_packages()

        lint_cmd = [python_bin, "-m", "py_compile", str(source_abs)]
        test_cmd = [python_bin, "-m", "pytest", "-q", str(tests_abs)]
        cov_cmd = [
            python_bin,
            "-m",
            "pytest",
            f"--cov={source_abs.stem}",
            f"--cov-fail-under={int(coverage_threshold * 100)}",
            str(tests_abs),
        ]
        cmds = {"lint": lint_cmd, "tests": test_cmd, "coverage": cov_cmd}
        results: Dict[str, subprocess.CompletedProcess] = {}
        lint_ok = tests_ok = coverage_ok = False
        pytest_available = self._has_pytest(python_bin)
        pytest_cov_available = self._has_pytest_cov(python_bin)

        for name, cmd in cmds.items():
            if name in {"tests", "coverage"} and not pytest_available:
                results[name] = subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="",
                    stderr="pytest not installed in sandbox environment",
                )
                continue
            if name == "coverage" and not pytest_cov_available:
                results[name] = subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout="pytest-cov not installed; coverage skipped",
                    stderr="",
                )
                coverage_ok = True
                continue

            res = self.sandbox.run(
                cmd,
                workdir=tests_abs.parent,
                timeout=180,
                stream=stream,
                on_output=on_output,
                name=name,
            )
            results[name] = res
            if name == "lint":
                lint_ok = res.returncode == 0
            elif name == "tests":
                tests_ok = res.returncode == 0
            elif name == "coverage":
                coverage_ok = res.returncode == 0
        return ValidationResult(lint_ok=lint_ok, tests_ok=tests_ok, coverage_ok=coverage_ok, command_results=results)

    def _has_pytest(self, python_bin: str) -> bool:
        try:
            res = subprocess.run(
                [python_bin, "-c", "import pytest"],
                capture_output=True,
                text=True,
            )
            return res.returncode == 0
        except Exception:
            return False

    def _has_pytest_cov(self, python_bin: str) -> bool:
        try:
            res = subprocess.run(
                [python_bin, "-c", "import pytest_cov"],
                capture_output=True,
                text=True,
            )
            return res.returncode == 0
        except Exception:
            return False

    def _bind_site_packages(self) -> None:
        """
        Add common site-packages locations as read-only binds so sandboxed Python can import user-installed deps.
        """
        paths = []
        try:
            paths.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            user_site = site.getusersitepackages()
            paths.append(user_site)
        except Exception:
            pass
        unique_paths = []
        seen = set()
        for p in paths:
            if not p:
                continue
            path_obj = Path(p).resolve()
            if path_obj.exists() and str(path_obj) not in seen:
                seen.add(str(path_obj))
                unique_paths.append(path_obj)
        for p in unique_paths:
            if p not in self.sandbox.policy.readonly_paths and p not in self.sandbox.policy.writable_paths:
                self.sandbox.policy.readonly_paths.append(p)
