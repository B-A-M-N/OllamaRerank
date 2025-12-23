"""
Legacy demo policy wrapper.

This preserves the current behavior by delegating to existing doc_quality
logic (item/domain/doc_kind/intent). It exists to keep the core generic
while providing the same outputs for the demo policy_id: "legacy_demo".
"""

from __future__ import annotations

from typing import Any

from rerank_service.policies import BasePolicy
from rerank_service.doc_quality import (
    detect_query_intent,
    detect_doc_kind,
    intent_fit_score,
    analyze_doc,
)


class LegacyDemoPolicy(BasePolicy):
    def policy_id(self) -> str:
        return "legacy_demo"

    def preprocess(self, query: str) -> dict:
        # For legacy, we just compute query intent once and pass it along.
        return {"query_intent": detect_query_intent(query)}

    def is_ambiguous(self, query: str) -> bool:
        # Keep existing ambiguity detection in the caller; legacy policy stays neutral.
        return False

    def doc_features(self, query: str, doc: str, ctx: dict) -> dict:
        quality = analyze_doc(doc)
        doc_kind = detect_doc_kind(doc, quality)
        q_intent = ctx.get("query_intent", "UNKNOWN")
        fit = intent_fit_score(q_intent, doc_kind)
        return {
            "quality": quality,
            "doc_kind": doc_kind,
            "query_intent": q_intent,
            "intent_fit": fit,
        }

