"""
Machinist: LLM autotooling pipeline with sandboxed execution and provenance-aware registry.
"""

from .pipeline import MachinistPipeline
from .registry import ToolRegistry
from .sandbox import BwrapSandbox
from .llm import LLMClient, OllamaClient, FallbackOllamaClient
from .config import MachinistConfig
from .explorer import LocalSystemExplorer, Artifact
from .validator import ValidationRunner, ValidationResult

from .auto_tool_creator import AutoToolCreator

__all__ = [
    "MachinistPipeline",
    "ToolRegistry",
    "BwrapSandbox",
    "LLMClient",
    "OllamaClient",
    "FallbackOllamaClient",
    "MachinistConfig",
    "LocalSystemExplorer",
    "Artifact",
    "ValidationRunner",
    "ValidationResult",
    "AutoToolCreator",
]
