import json
from typing import Any, IO, List


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.lower() in {"null", "none"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [v.strip().strip('"').strip("'") for v in value[1:-1].split(",") if v.strip()]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    if (value[0] == value[-1]) and value[0] in "\"'":
        return value[1:-1]
    return value


def _safe_load(text: str) -> dict:
    root: dict = {}
    stack: List[Any] = [(root, -1, "dict")]
    lines = text.splitlines()
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.lstrip()
        while stack and indent <= stack[-1][1]:
            stack.pop()
        parent, parent_indent, parent_type = stack[-1]
        if stripped.startswith("- "):
            content = stripped[2:].strip()
            if ":" in content:
                key, _, value = content.partition(":")
                item = {key.strip(): _parse_scalar(value)}
                parent.append(item)
                stack.append((item, indent, "dict"))
            else:
                parent.append(_parse_scalar(content))
            continue
        key, sep, rest = stripped.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = rest.strip()
        if not value:
            container: list = []
            if isinstance(parent, dict):
                parent[key] = container
            elif isinstance(parent, list):
                parent.append({key: container})
                container = parent[-1][key]
            stack.append((container, indent, "list"))
        else:
            parsed = _parse_scalar(value)
            if isinstance(parent, dict):
                parent[key] = parsed
            elif isinstance(parent, list):
                parent.append({key: parsed})
    return root


def safe_load(stream: IO[str]) -> dict:
    return _safe_load(stream.read())


def load(stream: IO[str]) -> dict:
    return safe_load(stream)
