CREATE_COMPOSITE_TOOL_PROMPT = """
You are an expert Python programmer. Your task is to create a single, standalone Python function from a `CompositionSpec` and the source code of the tools it uses.

**Composition Spec:**
```json
{{composition_spec_json}}
```

**Source Code of Component Tools:**
```python
{{component_tools_source}}
```

**Instructions:**
1.  Analyze the `CompositionSpec` to understand the workflow, including the sequence of steps and the data flow between them.
2.  Analyze the source code of the component tools to understand how to call them.
3.  Write a single Python function that implements the logic of the `CompositionSpec`.
4.  The new function should not call the component tools directly, but rather re-implement the logic within itself.
5.  The new function should have a signature that matches the `inputs` of the `CompositionSpec`.
6.  The new function should handle any loops or conditional logic defined in the `CompositionSpec`.
7.  The new function should be standalone and not have any external dependencies other than the Python standard library.
8.  You must return ONLY the generated Python code in a markdown code fence. Do not include any other text or explanations.
"""
