TEST_PROMPT = """
You are an expert software engineer specializing in writing comprehensive test suites.
Based on the provided ToolSpec contract, generate a complete pytest test file.

**ToolSpec Contract:**
```json
{{contract_json}}
```

**Test Skeleton (you must fill this in and add more tests):**
```python
{{test_skeleton}}
```

**Instructions:**
1.  Generate a complete pytest test file for the function described in the ToolSpec.
2.  The test file MUST include the exact import `{{required_import}}`.
3.  Write tests that cover:
    - The primary success case (the "happy path").
    - All failure modes specified in the `failure_modes` section of the contract. Each failure mode should have its own test function.
    - Edge cases (e.g., empty inputs, large inputs, files with special names, etc.).
4.  Use `pytest.raises` to test for expected exceptions.
5.  Use the `tmp_path` fixture for any filesystem operations to ensure tests are isolated.
6.  Do not include any code that is not part of the test file (e.g., the function implementation itself).

You must return ONLY the Python test code in a markdown code fence.
"""
