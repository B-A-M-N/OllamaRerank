from __future__ import annotations
import pytest
from pathlib import Path

from machinist.registry import ToolRegistry, ToolMetadata, ToolSpec
from machinist.templates import CompositionSpec, CompositionStep, StepBinding
from machinist.workflow import WorkflowEngine

# --- Mock Tools for Testing ---

def search_files_mock(root_dir: str, pattern: str) -> list[str]:
    """A mock for searching files."""
    print(f"Mock search in {root_dir} for {pattern}")
    if root_dir == "test_dir":
        return [f"test_dir/file1.txt", f"test_dir/file2.log"]
    return []

def copy_file_mock(src_path: str, dst_path: str) -> None:
    """A mock for copying files."""
    print(f"Mock copy from {src_path} to {dst_path}")
    # In a real test, you might touch the dst_path
    pass

@pytest.fixture
def populated_registry(tmp_path: Path) -> ToolRegistry:
    """Creates a ToolRegistry populated with mock executable tools for testing."""
    registry_path = tmp_path / "test_registry"
    registry = ToolRegistry(registry_path)

    # Manually create metadata and register it
    # 1. Search tool
    search_spec = ToolSpec("search_files_mock", "def search_files_mock(root_dir: str, pattern: str) -> list[str]:", "doc", {}, {}, [], True)
    search_meta = ToolMetadata("search_tool_id", "1.0", "now", search_spec, Path(""), Path(""), Path(""), {}, "", "", "", "fs.search_files.v1")
    registry.register(search_meta)
    registry._executable_cache["search_tool_id"] = search_files_mock

    # 2. Copy tool
    copy_spec = ToolSpec("copy_file_mock", "def copy_file_mock(src_path: str, dst_path: str) -> None:", "doc", {}, {}, [], True)
    copy_meta = ToolMetadata("copy_tool_id", "1.0", "now", copy_spec, Path(""), Path(""), Path(""), {}, "", "", "", "fs.copy.v1")
    registry.register(copy_meta)
    registry._executable_cache["copy_tool_id"] = copy_file_mock

    return registry


def test_workflow_engine_simple_execution(populated_registry: ToolRegistry):
    """Tests a simple linear workflow."""
    engine = WorkflowEngine(populated_registry)
    
    comp_spec = CompositionSpec(
        pipeline_id="test.simple_search",
        description="A simple test workflow.",
        inputs={"root": "path", "glob": "str"},
        steps=[
            CompositionStep(
                id="find",
                tool_id="fs.search_files.v1", # This is the template ID
                bind={
                    "root_dir": StepBinding(value="$root"),
                    "pattern": StepBinding(value="$glob"),
                },
                outputs={"files": "list[str]"}
            )
        ]
    )

    inputs = {"root": "test_dir", "glob": "*.txt"}
    final_context = engine.execute(comp_spec, inputs)

    assert final_context['find']['files'] == ["test_dir/file1.txt", "test_dir/file2.log"]

def test_workflow_engine_foreach_execution(populated_registry: ToolRegistry):
    """Tests a workflow with a foreach loop."""
    engine = WorkflowEngine(populated_registry)

    comp_spec = CompositionSpec(
        pipeline_id="test.find_and_copy",
        description="A test workflow with a loop.",
        inputs={"root_dir": "path", "dst_dir": "path"},
        steps=[
            CompositionStep(
                id="find_step",
                tool_id="fs.search_files.v1",
                bind={"root_dir": StepBinding(value="$root_dir"), "pattern": StepBinding(value="*")},
                outputs={"found": "list[str]"}
            ),
            CompositionStep(
                id="copy_step",
                tool_id="fs.copy.v1",
                foreach="$find_step.found",
                bind={
                    "src_path": StepBinding(value="$item"),
                    "dst_path": StepBinding(value="$dst_dir/copied_file"), # Simplified for test
                }
            )
        ]
    )

    inputs = {"root_dir": "test_dir", "dst_dir": "backup"}
    # We don't check the output of the copy mock, just that it runs without error
    engine.execute(comp_spec, inputs)


def test_workflow_engine_tool_not_found(populated_registry: ToolRegistry):
    """Tests that the engine raises an error if a tool for a template is not found."""
    engine = WorkflowEngine(populated_registry)

    comp_spec = CompositionSpec(
        pipeline_id="test.fail",
        description="A workflow that should fail.",
        steps=[
            CompositionStep(
                id="delete_step",
                tool_id="fs.delete.v1", # This template has no registered tool in our mock registry
                bind={"path": StepBinding(value="some_file.txt")}
            )
        ]
    )
    
    with pytest.raises(RuntimeError, match="No registered tool found for template 'fs.delete.v1'"):
        engine.execute(comp_spec, {})
