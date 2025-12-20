IMPLEMENT_PROMPT = """
You are an expert software engineer. Based on the provided ToolSpec contract, generate the complete, self-contained Python code for the function.

**ToolSpec Contract:**
```json
{{contract_json}}
```

**Instructions:**
1.  Implement the function `{{func_name}}` exactly as described in the ToolSpec.
2.  The implementation must be robust, production-quality, and secure.
3.  Include all necessary imports as specified in the contract. Do not add any extra imports.
4.  Handle all failure modes gracefully by raising the specified exceptions.
5.  Do not include any example usage, `if __name__ == "__main__"` block, or any other code outside of the function definition and its necessary imports.

You must return ONLY the Python code in a markdown code fence.

```python
# Your implementation here
```
"""
