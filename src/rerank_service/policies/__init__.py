"""
Policy registry and base interface.

The core rerank pipeline calls a Policy to perform domain/item-specific
logic (gating, doc_kind/intent classification, tie-break features,
ambiguous handling). This keeps the core generic and lets the demo ship
as a policy without hardcoding demo heuristics into the engine.
"""

from __future__ import annotations

import importlib
from typing import Dict, Type


class BasePolicy:
    """Minimal interface for policy plugins."""

    def preprocess(self, query: str):
        """Return any policy-specific context (may be None)."""
        return None

    def is_ambiguous(self, query: str) -> bool:
        """Optional: flag query as ambiguous/skip rerank."""
        return False

    def policy_id(self) -> str:
        """Return the policy identifier."""
        return "base"


# Registry mapping policy_id to import path:ClassName
POLICY_REGISTRY: Dict[str, str] = {
    # legacy demo policy that preserves current behavior
    "legacy_demo": "rerank_service.policies.legacy_demo:LegacyDemoPolicy",
    # generic policy placeholder (no domain gating)
    "generic": "rerank_service.policies.generic:GenericPolicy",
}


def load_policy(policy_id: str) -> BasePolicy:
    """Instantiate a policy by id using the registry."""
    target = POLICY_REGISTRY.get(policy_id)
    if not target:
        raise ValueError(f"Unknown policy_id '{policy_id}'. Known: {list(POLICY_REGISTRY)}")
    module_path, cls_name = target.split(":")
    module = importlib.import_module(module_path)
    cls: Type[BasePolicy] = getattr(module, cls_name)
    return cls()

