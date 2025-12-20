SPEC_PROMPT = """
You are an expert software engineer specializing in creating self-contained, production-quality Python tools.
Your task is to generate a JSON specification (ToolSpec) for a Python function based on a natural language goal.

The ToolSpec format is a JSON object with the following keys:
- "name" (str): A valid, descriptive Python function name (snake_case).
- "signature" (str): The full Python function signature, including type hints.
- "docstring" (str): A comprehensive docstring explaining what the function does, its parameters, and what it returns.
- "imports" (list[str]): A list of standard Python libraries required (e.g., "os", "re"). No external libraries are allowed.
- "inputs" (dict[str, str]): A dictionary where keys are parameter names and values are their descriptions.
- "outputs" (dict[str, str]): A dictionary describing the function's return value(s).
- "failure_modes" (list[dict[str, str]]): A list of objects, each describing a failure case with "exception" and "reason" keys.
- "deterministic" (bool): True if the function always produces the same output for the same input, False otherwise.

Constraints:
- The function must be self-contained and use only the specified imports.
- The function name must be a valid Python identifier.
- The signature must be syntactically correct Python.
- The signature's parameters must exactly match the keys in the "inputs" dictionary.

You must return ONLY the JSON object in a markdown code fence. Do not include any other text, explanations, or wrappers.

Example:
Goal: "a tool to read a file"
```json
{
  "name": "read_file",
  "signature": "def read_file(path: str) -> str:",
  "docstring": "Reads the entire content of a file and returns it as a string.",
  "imports": ["os"],
  "inputs": {
    "path": "The path of the file to read."
  },
  "outputs": {
    "content": "The string content of the file."
  },
  "failure_modes": [
    {
      "exception": "FileNotFoundError",
      "reason": "The specified file path does not exist."
    }
  ],
  "deterministic": true
}
```
"""

SPEC_PROMPT_FROM_TEMPLATE = """
You are an expert software engineer specializing in creating self-contained, production-quality Python tools.
Your task is to generate a JSON specification (ToolSpec) for a Python function based on a natural language goal and a "Pseudo-Spec Template".
The template provides strict constraints that you MUST follow.

**Pseudo-Spec Template (Constraints):**
```json
{{template_json}}
```

**Your Goal:**
"{{goal}}"

**Instructions:**
1.  Read the user's goal and the provided pseudo-spec template.
2.  Generate a complete `ToolSpec` JSON object that fulfills the goal while strictly adhering to ALL constraints from the template.
3.  The `name` of the function should be descriptive of the goal, but consistent with the template's intent (e.g., for a 'copy' intent, a name like `duplicate_config_file` is good).
4.  The `signature` and `inputs` must match the `param_skeletons` in the template.
5.  The `imports` must only contain modules from the template's `allowed_imports`.
6.  The `failure_modes` must include all `base_failure_modes` from the template.
7.  The `docstring` must be well-written and accurately describe the function, its parameters, and what it returns. The language used in the docstring should not contradict the `forbidden_verbs` from the template.

You must return ONLY the final `ToolSpec` JSON object in a markdown code fence. Do not include any other text, explanations, or wrappers.
"""
