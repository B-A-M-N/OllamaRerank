from pathlib import Path
import pytest

from build_toolchain import build_toolchain


def test_happy_path(tmp_path: Path):
    config = {
        "version": 1,
        "components": ["component1", "component2"],
        "dependencies": {"dep1": "version1", "dep2": "version2"}
    }
    
    path1 = tmp_path / "comp1"
    path2 = tmp_path / "comp2"
    path1.mkdir()
    path2.mkdir()

    toolchain_config = build_toolchain(config, [str(path1), str(path2)])
    assert "toolchain" in toolchain_config
    assert "required_tools" in toolchain_config
    assert len(toolchain_config["toolchain"]) == 2
    assert len(toolchain_config["required_tools"]) == 2

def test_missing_component_paths():
    config = {
        "version": 1,
        "components": ["missing_comp"],
        "dependencies": {"dep1": "version1", "dep2": "version2"}
    }
    with pytest.raises(ValueError):
        build_toolchain(config, ["path/to/comp"])

def test_invalid_configuration():
    invalid_config = {
        "version": 1,
        "components": ["component"],
        "dependencies": {"dep": "v"}
    }
    with pytest.raises(ValueError):
        build_toolchain(invalid_config, ["path/to/comp"])