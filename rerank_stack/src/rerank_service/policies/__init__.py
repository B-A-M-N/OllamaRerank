"""
Policy registry and base interface.

The core rerank pipeline calls a Policy to perform domain/item-specific
logic (gating, doc_kind/intent classification, tie-break features,
ambiguous handling). This keeps the core generic and lets the demo ship
as a policy without hardcoding demo heuristics into the engine.
"""

from __future__ import annotations

import importlib
from typing import Dict, Type, Optional, Protocol, Any, List, Tuple

from rerank_service.schemas import Document, Result, Options


class BasePolicy:
    """Minimal interface for policy plugins."""

    def preprocess(self, query: str):
        """Return any policy-specific context (may be None)."""
        return None

    def build_context(self, query: str, policy_id_override: Optional[str] = None):
        """Return a policy-specific context object (may be None)."""
        return None

    def precompute_facts(self, query: str, documents: List[Document]) -> Dict[int, dict]:
        """Return facts keyed by document.original_index."""
        return {}

    def clamp_bucket(self, score: int, facts: dict) -> int:
        """Optional clamp of score bucket based on facts."""
        return score

    def rank(
        self,
        query: str,
        documents: List[Document],
        scored_docs: List[Tuple[int, int, int, int]],
        similarities_by_req: List[Optional[float]],
        facts_map: Dict[int, dict],
        debug: bool,
        options: Options,
    ) -> List[Result]:
        """Return ranked results with optional facts."""
        raise NotImplementedError

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
    "generic_v1": "rerank_service.policies.generic:GenericPolicy",
}


def load_policy(policy_id: Optional[str]) -> BasePolicy:
    """Instantiate a policy by id using the registry."""
    pid = (policy_id or "legacy_demo").strip()
    if pid.startswith("howto_"):
        pid = "legacy_demo"
    if pid not in POLICY_REGISTRY and pid.endswith("_v1"):
        pid = pid[:-3]
    target = POLICY_REGISTRY.get(pid)
    if not target:
        raise ValueError(f"Unknown policy_id '{pid}'. Known: {list(POLICY_REGISTRY)}")
    module_path, cls_name = target.split(":")
    module = importlib.import_module(module_path)
    cls: Type[BasePolicy] = getattr(module, cls_name)
    return cls()
