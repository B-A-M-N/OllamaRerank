from __future__ import annotations
import re
import ast

def add_missing_imports(code: str) -> str:
    """
    Heuristically finds common missing imports and adds them.
    This is a simple, regex-based approach to fix frequent LLM omissions.
    """
    # List of (module, usage_pattern)
    known_imports = [
        ("os", r"\bos\.|\bos\("),
        ("shutil", r"\bshutil\.|\bshutil\("),
        ("errno", r"\berrno\.|\berrno\("),
        ("pytest", r"\bpytest\.|\bpytest\("),
        ("re", r"\bre\.|\bre\("),
    ]

    # Find existing imports
    existing_imports = set(re.findall(r"^\s*import\s+([^\s,]+)", code, re.MULTILINE))
    existing_imports.update(re.findall(r"^\s*from\s+([^\s,]+)\s+import", code, re.MULTILINE))

    # Determine which imports are missing
    imports_to_add = []
    for module, pattern in known_imports:
        if module not in existing_imports:
            if re.search(pattern, code):
                imports_to_add.append(f"import {module}")

    if not imports_to_add:
        return code

    # Add new imports to the top of the code
    new_imports_str = "\n".join(imports_to_add)
    return f"{new_imports_str}\n{code}"

def remove_unused_typing_imports(code: str) -> str:
    """
    Removes unused imports from the 'typing' module.
    """
    try:
        tree = ast.parse(code)
        
        used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        
        lines = code.splitlines()
        new_lines = []

        for line in lines:
            if line.strip().startswith('from typing import'):
                imports = [imp.strip() for imp in line.replace('from typing import', '').split(',')]
                used_imports = [imp for imp in imports if imp in used_names]

                if used_imports:
                    new_lines.append(f"from typing import {', '.join(used_imports)}")
            else:
                new_lines.append(line)
        
        return '\n'.join(new_lines)

    except SyntaxError:
        return code
