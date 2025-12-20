from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional


class LLMClient(ABC):
    """
    Small abstraction so we can swap Ollama models without touching the pipeline.
    Implementations should be pure (no side effects) and deterministic when temperature=0.
    """

    @abstractmethod
    def complete(self, system: str, prompt: str, *, temperature: float = 0.0, max_tokens: int | None = None) -> str:
        ...

    @abstractmethod
    def chat(self, messages: List[dict[str, str]], *, temperature: float = 0.0, max_tokens: int | None = None) -> str:
        ...

    def chat_with_tools(
        self,
        messages: List[dict[str, str]],
        *,
        tools: List[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Optional: returns the raw response dict for tool-calling models.
        Default implementation falls back to normal chat() as plain text.
        """
        # This fallback provides a basic text response if tool-calling is not supported.
        text = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return {"message": {"role": "assistant", "content": text}} # tool_calls will be missing

    @abstractmethod
    def model_name(self) -> str:
        ...

    def stream_complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        on_token: Optional[callable] = None,
    ) -> str:
        """
        Optional streaming interface. Default implementation falls back to complete().
        Implementations should call on_token(token_text) as chunks arrive.
        """
        return self.complete(system, prompt, temperature=temperature, max_tokens=max_tokens)


def render_prompt(template: str, **kwargs: Any) -> str:
    """Safe-ish prompt renderer; only formats known placeholders."""
    # First, escape all braces
    template = template.replace("{", "{{").replace("}", "}}")
    # Then, un-escape the intended placeholders
    for key in kwargs:
        template = template.replace("{{" + key + "}}", "{" + key + "}")
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f"Missing prompt variable: {missing}") from exc


DEFAULT_SYSTEM_PROMPT = (
    "You are Machinist, a strict tool builder. "
    "You produce deterministic code that respects the provided contract and sandbox policy. "
    "Do not invent APIs; be explicit about assumptions."
)