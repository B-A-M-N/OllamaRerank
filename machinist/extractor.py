from __future__ import annotations
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional

from .registry import ToolSpec, ToolRegistry
from .templates import PseudoSpecTemplate, CompositionSpec, CompositionStep, StepBinding, TEMPLATE_REGISTRY

class SideEffectAnalyzer(ast.NodeVisitor):
    """
    Scans an AST to infer side effects like file system or network access.
    """
    def __init__(self):
        self.side_effects: set[str] = set()

    def visit_Call(self, node: ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in ["open", "write", "write_text", "remove", "unlink", "move", "copy", "rmdir", "mkdir", "makedirs"]:
            is_write = False
            if func_name == "open":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and any(c in arg.value for c in 'wa+x'):
                        is_write = True
                for kw in node.keywords:
                    if kw.arg == 'mode' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str) and any(c in kw.value.value for c in 'wa+x'):
                        is_write = True
            if is_write or func_name != "open":
                 self.side_effects.add("filesystem_write")

        if func_name in ["open", "read", "read_text", "listdir", "walk", "stat"]:
             self.side_effects.add("filesystem_read")

        if func_name in ["get", "post", "put", "delete", "request", "urlopen"]:
            self.side_effects.add("network")
        
        if func_name in ["run", "call", "check_call", "system"]:
            self.side_effects.add("process")

        self.generic_visit(node)

class ExceptionAnalyzer(ast.NodeVisitor):
    """
    Scans an AST to find all explicitly raised exceptions.
    """
    def __init__(self):
        self.raised_exceptions: set[str] = set()

    def visit_Raise(self, node: ast.Raise):
        if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
            self.raised_exceptions.add(node.exc.func.id)
        self.generic_visit(node)


def _spec_from_function_node(func_node: ast.FunctionDef, source_code: str, module_imports: List[str]) -> ToolSpec:
    """
    Private helper to generate a ToolSpec from a function's AST node.
    """
    signature = ast.get_source_segment(source_code, func_node, padded=False).split(':\n')[0] + ':'
    docstring = ast.get_docstring(func_node) or ""

    exception_analyzer = ExceptionAnalyzer()
    exception_analyzer.visit(func_node)
    failure_modes = [{"exception": name, "reason": "Inferred from code."} for name in sorted(exception_analyzer.raised_exceptions)]

    side_effect_analyzer = SideEffectAnalyzer()
    side_effect_analyzer.visit(func_node)
    deterministic = not bool(side_effect_analyzer.side_effects)

    inputs = {arg.arg: "" for arg in func_node.args.args}
    outputs = {"return_value": ""} if func_node.returns else {}

    return ToolSpec(
        name=func_node.name,
        signature=signature,
        docstring=docstring,
        imports=module_imports,
        inputs=inputs,
        outputs=outputs,
        failure_modes=failure_modes,
        deterministic=deterministic,
    )


def spec_from_existing_function(file_path: str, func_name: str, registry: ToolRegistry) -> Optional[ToolSpec]:
    """
    Outputs a ToolSpec for a single function in a file and caches it.
    """
    try:
        p = Path(file_path)
        source_code = p.read_text(encoding="utf-8")
        tree = ast.parse(source_code)
    except (FileNotFoundError, SyntaxError) as e:
        print(f"Error reading or parsing file {file_path}: {e}")
        return None

    func_node = None
    module_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            func_node = node
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_imports.append(node.module)

    if not func_node:
        print(f"Function '{func_name}' not found in {file_path}")
        return None
    
    spec = _spec_from_function_node(func_node, source_code, sorted(list(set(module_imports))))
    
    # Cache the extracted spec
    cache_id = registry.cache_spec(spec, {"source": "extractor", "file": file_path, "function": func_name})
    print(f"  -> Cached extracted spec for {func_name} with ID: {cache_id}")
    
    return spec

def specs_from_directory(directory_path: str, registry: ToolRegistry) -> List[ToolSpec]:
    """
    Finds all Python files in a directory and extracts a ToolSpec from each 'tool' function, caching them.
    """
    specs = []
    root = Path(directory_path)
    if not root.is_dir():
        print(f"Error: Provided path '{directory_path}' is not a directory.")
        return specs

    for py_file in root.rglob("*.py"):
        print(f"Scanning file: {py_file}")
        try:
            source_code = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source_code)
            
            module_imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_imports.append(node.module)
            
            unique_imports = sorted(list(set(module_imports)))

            for node in tree.body: # Iterate over top-level nodes only
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    print(f"  -> Found tool function: {node.name}")
                    spec = _spec_from_function_node(node, source_code, unique_imports)
                    
                    # Cache the extracted spec
                    cache_id = registry.cache_spec(spec, {"source": "extractor", "file": str(py_file), "function": node.name})
                    print(f"    -> Cached extracted spec with ID: {cache_id}")
                    
                    specs.append(spec)
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"  -> Skipping file {py_file} due to parsing error: {e}")
            continue
            
    return specs

def comp_spec_from_existing_script(script_path: str, registry: ToolRegistry) -> Optional[CompositionSpec]:
    """
    Derives a CompositionSpec from an existing Python script and caches it.
    """
    try:
        p = Path(script_path)
        source_code = p.read_text(encoding="utf-8")
        tree = ast.parse(source_code)
    except (FileNotFoundError, SyntaxError) as e:
        print(f"Error reading or parsing script {script_path}: {e}")
        return None

    # ... (rest of the implementation for comp spec extraction)
    # For now, just returning a placeholder and caching it
    
    # This is a placeholder for the real implementation
    comp_spec = CompositionSpec(pipeline_id=p.stem, description=f"Placeholder for {p.name}", steps=[])
    
    if comp_spec:
        cache_id = registry.cache_spec(comp_spec, {"source": "extractor", "script": script_path})
        print(f"Cached extracted composition spec with ID: {cache_id}")
        return comp_spec

    return None