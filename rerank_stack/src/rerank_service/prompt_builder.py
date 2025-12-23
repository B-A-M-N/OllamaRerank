# rerank_stack/src/rerank_service/prompt_builder.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .policy import PolicyContext


def _as_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=( ",", ":"))


def build_rerank_prompt_single_doc(query: str, document_text: str, context: PolicyContext, facts: Dict[str, Any]) -> str:
    """
    Model MUST output: {"score": 0|1|2|3}
    Score is a BUCKET INDEX (not legacy 0/3/7/10).
      3 = same item + actionable steps (2+ concrete actions)
      2 = same item + partial/diagnostic/overview (<=1 concrete action OR mostly diagnosis)
      1 = same domain but different item (adjacent)
      0 = unrelated / different domain
    """

    rules = (
        "You are a deterministic relevance bucketer.\n"
        "Output EXACTLY one JSON object and nothing else:\n"
        '{"score":0} or {"score":1} or {"score":2} or {"score":3}\n\n'
        "Use the provided FACTS. Do NOT infer item identity yourself.\n\n"
        "BUCKETS:\n"
        "3: SAME ITEM and gives actionable fix steps (2+ concrete actions like turn off/remove/replace/install/test).\n"
        "2: SAME ITEM but only partial help (diagnosis, overview, or <=1 concrete action).\n"
        "1: DIFFERENT ITEM but same domain (adjacent topic).\n"
        "0: Different domain / unrelated.\n\n"
        "DETERMINISTIC RULES:\n"
        "- If FACTS.query_item is null (generic query), score based on domain only:\n"
        "  * score=2 if FACTS.domain_match is true AND the doc is about maintenance/repair concepts.\n"
        "  * score=1 if FACTS.domain_match is true but only loosely related.\n"
        "  * score=0 if FACTS.domain_match is false.\n"
        "- If FACTS.query_item is not null (specific query):\n"
        "  * If FACTS.same_item is true: score must be 2 or 3.\n"
        "  * If FACTS.same_item is false:\n"
        "      - score=1 if FACTS.domain_match is true\n"
        "      - score=0 if FACTS.domain_match is false\n"
    )

    # Few-shot: we keep them, but they now demonstrate using FACTS fields.
    # Note: we re-use policy.few_shot query/doc/score but we “wrap” them into the same structure.
    examples: List[str] = []
    for shot in context.policy.few_shot:
        shot_facts = context.compute_identity_facts(query=shot["query"], doc=shot["doc"])
        examples.append(
            "EXAMPLE\n"
            f"QUERY: {shot['query']}\n"
            f"DOCUMENT: {shot['doc']}\n"
            f"FACTS: {_as_json(shot_facts)}\n"
            f"OUTPUT: {_as_json({'score': context.to_bucket_index(shot['score'])})}\n"
        )

    examples_section = "\n---\n".join(examples)

    return (
        f"{rules}\n"
        f"{examples_section}\n"
        "\n---\n"
        f"QUERY: {query}\n"
        f"DOCUMENT: {document_text}\n"
        f"FACTS: {_as_json(facts)}\n"
        "OUTPUT:"
    )


def build_retry_score_prompt(
    query: str,
    document_text: str,
    previous_output: str,
    context: PolicyContext,
    facts: Dict[str, Any],
) -> str:
    """
    Retry prompt is ultra-minimal and still JSON-only.
    """
    return (
        "Output EXACTLY one JSON object and nothing else:\n"
        '{"score":0} or {"score":1} or {"score":2} or {"score":3}\n\n'
        f"FACTS: {_as_json(facts)}\n"
        "OUTPUT:"
    )
