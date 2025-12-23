from __future__ import annotations

"""
Generic document-quality signals (core).

This module intentionally exposes ONLY generic signals and helpers. Any
policy-specific logic (intent, doc_kind, domain rules, gating, postprocess)
must live under rerank_service.policies.
"""

from .core.quality import (
    QualitySignals,
    analyze_signals,
    annotate_corpus,
)


# Backward-compatible alias for legacy callers.
def analyze_doc(text: str) -> QualitySignals:
    return analyze_signals(text)

