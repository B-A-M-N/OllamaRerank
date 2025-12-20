import os
from typing import List

def build_toolchain(config: dict, tool_paths: List[str]) -> dict:
    """Builds a tool chain configuration based on given parameters and paths."""
    if not config or not tool_paths:
        raise ValueError("Invalid configuration dictionary or empty list of tool paths")

    toolchain_config = {
        "toolchain": {},
        "required_tools": set()
    }

    for path in tool_paths:
        if os.path.exists(path):
            tool_name = os.path.basename(path)
            toolchain_config["toolchain"][tool_name] = {"path": path}
            toolchain_config["required_tools"].add(tool_name)
        else:
            raise ValueError(f"Tool path {path} does not exist")

    return toolchain_config