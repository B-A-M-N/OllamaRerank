GENERATE_AUTONOMOUS_GOAL_PROMPT = """
You are an expert system architect. Your task is to propose a new, useful tool that can be created by composing the existing tools.

**Existing Tools:**
```json
{{existing_tools_json}}
```

**Instructions:**
1.  Analyze the existing tools and their capabilities.
2.  Propose a high-level goal for a new tool that would be a valuable addition. The new tool should combine the functionality of two or more existing tools to achieve a more complex task.
3.  The goal should be a single, concise sentence.
4.  Do not propose a tool that is too similar to an existing tool.
5.  Do not propose a tool that cannot be built from the existing tools.

**Output Format:**
You must return ONLY the goal as a single line of text. Do not include any other text or explanations.
"""
