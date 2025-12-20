IMPROVE_SPEC_PROMPT = """
You are an expert software architect. Your task is to improve an existing ToolSpec based on a user's improvement goal.
You will be given the current `ToolSpec` and a natural language goal for how to improve it.
You must return a new, improved `ToolSpec` in a valid JSON format.

**Current ToolSpec:**
```json
{{current_spec_json}}
```

**User's Improvement Goal:**
"{{improvement_goal}}"

**Instructions:**
1.  Analyze the current spec and the user's improvement goal.
2.  Modify the `ToolSpec` to meet the goal. This might involve:
    - Adding, removing, or changing parameters in the `signature` and `inputs`.
    - Updating the `docstring` to reflect the new behavior.
    - Adding new `imports` if required (standard library only).
    - Adding new `failure_modes`.
    - Changing the `name` if the tool's core identity changes.
3.  The new `ToolSpec` must be a complete and valid JSON object.
4.  Do not change the fundamental intent of the tool unless the goal explicitly asks for it. For example, do not change a "copy" tool into a "move" tool.

**Output Format:**
You must return ONLY the new, improved `ToolSpec` JSON object in a markdown code fence. Do not include any other text or explanations.
"""
