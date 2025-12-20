from __future__ import annotations

from typing import Any, List, Optional

from .base import LLMClient


class OpenAICompatibleClient(LLMClient):
    """
    Client for a generic OpenAI-compatible server.
    """

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    def model_name(self) -> str:
        return self._model

    def complete(
        self, system: str, prompt: str, *, temperature: float = 0.0, max_tokens: int | None = None
    ) -> str:
        raise NotImplementedError("This client is a stub and is not yet implemented.")

    def chat(
        self, messages: List[dict[str, str]], *, temperature: float = 0.0, max_tokens: int | None = None
    ) -> str:
        raise NotImplementedError("This client is a stub and is not yet implemented.")

    def chat_with_tools(
        self,
        messages: List[dict[str, Any]],
        *,
        tools: List[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("This client is a stub and is not yet implemented.")


class HuggingFaceClient(LLMClient):
    """
    Client for a Hugging Face model.
    """

    def __init__(self, model: str, api_key: str) -> None:
        self._model = model
        self._api_key = api_key

    def model_name(self) -> str:
        return self._model

    def complete(
        self, system: str, prompt: str, *, temperature: float = 0.0, max_tokens: int | None = None
    ) -> str:
        raise NotImplementedError("This client is a stub and is not yet implemented.")

    def chat(
        self, messages: List[dict[str, str]], *, temperature: float = 0.0, max_tokens: int | None = None
    ) -> str:
        raise NotImplementedError("This client is a stub and is not yet implemented.")

    def chat_with_tools(
        self,
        messages: List[dict[str, Any]],
        *,
        tools: List[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("This client is a stub and is not yet implemented.")


class VLLMClient(LLMClient):
    """
    Client for a vLLM deployment.
    """

    def __init__(self, model: str, host: str, port: int) -> None:
        self._model = model
        self._host = host
        self._port = port

    def model_name(self) -> str:
        return self._model

    def complete(
        self, system: str, prompt: str, *, temperature: float = 0.0, max_tokens: int | None = None
    ) -> str:
        raise NotImplementedError("This client is a stub and is not yet implemented.")

    def chat(
        self, messages: List[dict[str, str]], *, temperature: float = 0.0, max_tokens: int | None = None
    ) -> str:
        raise NotImplementedError("This client is a stub and is not yet implemented.")

    def chat_with_tools(
        self,
        messages: List[dict[str, Any]],
        *,
        tools: List[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("This client is a stub and is not yet implemented.")
