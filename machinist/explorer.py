from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import MachinistConfig


@dataclass
class Artifact:
    path: Path
    artifact_type: str
    description: str
    metadata: Dict[str, str]
    discovered_at: str


class LocalSystemExplorer:
    """
    Workspace-scoped explorer for discovering relevant artifacts.
    Will not traverse outside configured exploration paths and skips secret-like globs.
    """

    def __init__(self, config: MachinistConfig) -> None:
        self.config = config
        self.skip_dirs = {"/proc", "/sys", "/dev", "/etc", "/bin", "/sbin", "/var", "/boot", "/root"}

    def scan(self, base_paths: Optional[List[Path]] = None) -> List[Artifact]:
        if not self.config.enable_exploration:
            return []
        paths = base_paths or self.config.resolved_paths()
        artifacts: List[Artifact] = []
        for base in paths:
            if not self._is_safe_path(base):
                continue
            artifacts.extend(self._scan_directory(base, depth=0))
        return artifacts

    def _is_safe_path(self, path: Path) -> bool:
        abs_path = path.resolve()
        for skip in self.skip_dirs:
            if str(abs_path).startswith(skip):
                return False
        for allowed in self.config.resolved_paths():
            if str(abs_path).startswith(str(allowed)):
                return True
        return False

    def _scan_directory(self, directory: Path, depth: int) -> List[Artifact]:
        if depth >= self.config.max_exploration_depth:
            return []
        artifacts: List[Artifact] = []
        try:
            for root, dirs, files in os.walk(directory):
                # prevent descent into disallowed dirs
                dirs[:] = [
                    d
                    for d in dirs
                    if self._is_safe_path(Path(root) / d)
                ]
                for filename in files:
                    if any(fnmatch.fnmatch(filename, pattern) for pattern in self.config.exploration_ignore_globs):
                        continue
                    filepath = Path(root) / filename
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in self.config.exploration_file_types:
                        artifact = self._analyze_file(filepath, ext.lower())
                        if artifact:
                            artifacts.append(artifact)
                if depth >= self.config.max_exploration_depth - 1:
                    dirs.clear()
        except Exception:
            return artifacts
        return artifacts

    def _analyze_file(self, filepath: Path, file_ext: str) -> Optional[Artifact]:
        if not os.access(filepath, os.R_OK):
            return None
        size = filepath.stat().st_size
        if size > 10 * 1024 * 1024:
            return None
        artifact_type = "unknown"
        description = filepath.name
        metadata: Dict[str, str] = {
            "size": str(size),
            "extension": file_ext,
            "last_modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
        }
        if file_ext == ".csv":
            artifact_type = "csv_data"
        elif file_ext == ".json":
            artifact_type = "json_data"
        elif file_ext == ".py":
            artifact_type = "code"
        return Artifact(
            path=filepath,
            artifact_type=artifact_type,
            description=description,
            metadata=metadata,
            discovered_at=datetime.utcnow().isoformat(),
        )
