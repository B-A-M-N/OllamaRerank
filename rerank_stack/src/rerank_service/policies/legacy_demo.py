"""
Legacy demo policy wrapper.

This preserves the current behavior by delegating to existing doc_quality
logic (item/domain/doc_kind/intent). It exists to keep the core generic
while providing the same outputs for the demo policy_id: "legacy_demo".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rerank_service.policies import BasePolicy
from rerank_service.policies.legacy_demo_logic import (
    rerank_postprocess,
    _is_ambiguous_query,
)
from rerank_service.policy import policy_manager, build_policy_context, PolicyContext
from rerank_service.schemas import Document, Result, Options


class LegacyDemoPolicy(BasePolicy):
    def policy_id(self) -> str:
        return "legacy_demo"

    def __init__(self) -> None:
        self._policy = None
        self._context: Optional[PolicyContext] = None

    def build_context(self, query: str, policy_id_override: Optional[str] = None) -> PolicyContext:
        policy = policy_manager.select_policy(policy_id_override, query)
        self._policy = policy
        self._context = build_policy_context(policy)
        return self._context

    def preprocess(self, query: str) -> dict:
        # Legacy policy relies on PolicyContext for intent/doc_kind.
        return {}

    def is_ambiguous(self, query: str) -> bool:
        if not self._context:
            return False
        return _is_ambiguous_query(query, self._context)

    def precompute_facts(self, query: str, documents: List[Document]) -> Dict[int, dict]:
        if not self._context:
            self._context = self.build_context(query)
        facts_map: Dict[int, dict] = {}
        for doc in documents:
            facts_map[doc.original_index] = self._context.compute_identity_facts(
                query=query,
                doc=doc.text,
            )
        return facts_map

    def clamp_bucket(self, score: int, facts: dict) -> int:
        # Hard domain gate
        if not facts.get("domain_match", False):
            return 0

        # Item identity gate: if query names an item and doc does NOT match it, clamp to lowest bucket (0)
        if facts.get("query_item") is not None and not facts.get("same_item", False):
            return 0

        # Identity passed: allow the model’s full bucket range (no extra clamp)
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
        if not self._context:
            self._context = self.build_context(query)

        original_docs_map = {doc.original_index: doc.text for doc in documents}
        mode = "AMBIGUOUS" if self.is_ambiguous(query) else "SPECIFIC"

        # Clamp buckets and apply floor.
        clamped_scored_documents: List[Tuple[int, int, int, int]] = []
        for req_idx, orig_idx, score, tie_breaker in scored_docs:
            facts = facts_map.get(orig_idx, {})
            if mode == "AMBIGUOUS":
                new_score = score
            else:
                new_score = self.clamp_bucket(score, facts)
                if new_score != score:
                    print(
                        "### ITEM_CLAMP_ACTIVE ### "
                        f"Query: {query}, Doc: {orig_idx}. Score: {score} -> {new_score}"
                    )
                if new_score == 0 and facts.get("same_domain") and facts.get("same_item"):
                    print(
                        "### FLOOR_ACTIVE ### "
                        f"Query: {query}, Doc: {orig_idx}. same_domain+same_item but score=0; flooring to 1."
                    )
                    new_score = 1
            clamped_scored_documents.append((req_idx, orig_idx, new_score, tie_breaker))

        clamped_scored_documents.sort(key=lambda x: (x[2], x[3], -x[0]), reverse=True)

        docs_for_postprocess = [
            original_docs_map[orig_idx] for _, orig_idx, _, _ in clamped_scored_documents
        ]
        scores_for_postprocess = [score for _, _, score, _ in clamped_scored_documents]
        tie_breakers_for_postprocess = [tb for _, _, _, tb in clamped_scored_documents]

        max_rank = max(1, len(clamped_scored_documents) - 1)
        original_similarities: List[float] = []
        for req_idx, _, _, _ in clamped_scored_documents:
            sim = similarities_by_req[req_idx] if req_idx < len(similarities_by_req) else None
            if sim is not None:
                original_similarities.append(float(sim))
            else:
                original_similarities.append(1.0 - (req_idx / max_rank))

        policy_name = getattr(self._policy, "name", self.policy_id())

        def _zero_tier_from_facts(facts: dict) -> int:
            zr = facts.get("zero_reason")
            same_domain = facts.get("same_domain", False)
            same_item = facts.get("same_item", False)
            if zr == "item_mismatch":
                return 0
            if zr == "no_item_found":
                return 1 if (same_domain or same_item) else 2
            return 2

        postprocessed_results = rerank_postprocess(
            query=query,
            docs=docs_for_postprocess,
            rerank_scores=scores_for_postprocess,
            original_similarities=original_similarities,
            tie_breakers=tie_breakers_for_postprocess,
            context=self._context,
        )

        ranked_results: List[Result] = []
        for relative_idx, final_score, rank_score, _quality, query_intent, doc_kind, intent_fit in postprocessed_results:
            req_idx, original_corpus_index, _score, _tie_breaker = clamped_scored_documents[relative_idx]
            facts = facts_map.get(original_corpus_index, {}).copy() if facts_map.get(original_corpus_index) else {}
            facts["final_rank_score"] = rank_score
            facts["rank_score"] = rank_score  # backward compatibility
            facts["query_intent"] = query_intent
            facts["doc_kind"] = doc_kind
            facts["intent_fit"] = intent_fit
            facts["mode"] = mode
            if final_score == 0:
                if not facts.get("same_domain", False) and mode != "AMBIGUOUS":
                    facts["zero_reason"] = "domain_mismatch"
                elif facts.get("query_item") is not None and facts.get("same_item") is False and mode != "AMBIGUOUS":
                    facts["zero_reason"] = "item_mismatch"
                elif not facts.get("doc_items") and mode == "SPECIFIC":
                    facts["zero_reason"] = "no_item_found"
                    # Soft-keep: demote instead of hard-reject for missing item, to keep adjacent-but-helpful docs.
                    score_raw = 0.1 * (facts.get("intent_fit") or 0.0)
                    final_score = score_raw
                    score_norm = max(0.0, min(1.0, score_raw / 3.0))
                else:
                    facts["zero_reason"] = "low_signal"

            zero_tier = _zero_tier_from_facts(facts)
            tier = _score
            score_raw = float(final_score)
            score_norm = max(0.0, min(1.0, score_raw / 3.0))
            decision = "ACCEPT" if tier > 0 else f"REJECT_TIER_{zero_tier}"
            pre_rank_score = rank_score
            rank_score_final = pre_rank_score if decision == "ACCEPT" else 0.0
            rank_score_reason = "accepted" if decision == "ACCEPT" else "rejected"
            facts.update(
                {
                    "policy_id": policy_name,
                    "tier": tier,
                    "score_raw": score_raw,
                    "score_norm": score_norm,
                    "decision": decision,
                    "mode": mode,
                    "pre_rank_score": pre_rank_score,
                    "rank_score": rank_score_final,
                    "rank_score_reason": rank_score_reason,
                }
            )

            ranked_results.append(
                Result(
                    index=relative_idx,
                    original_corpus_index=original_corpus_index,
                    score=score_raw,
                    pre_rank_score=pre_rank_score,
                    rank_score=rank_score_final,
                    rank_score_reason=rank_score_reason,
                    score_raw=score_raw,
                    score_norm=score_norm,
                    tier=tier,
                    decision=decision,
                    policy_id=policy_name,
                    facts=facts,
                )
            )

        # Build a lookup for similarities by original index.
        sim_by_orig = {
            doc.original_index: (doc.similarity if doc.similarity is not None else 0.0)
            for doc in documents
        }

        def _safe_sim(orig_idx: int) -> float:
            value = sim_by_orig.get(orig_idx, 0.0)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
            return max(0.0, value)

        # Ambiguous: soft floor to avoid hard zero collapse.
        if mode == "AMBIGUOUS":
            for res in ranked_results:
                if (res.score or 0) <= 0:
                    sim_val = _safe_sim(res.original_corpus_index)
                    soft_floor = 0.12 * sim_val
                    if soft_floor > (res.score or 0):
                        res.score = soft_floor
                        res.score_norm = max(res.score_norm or 0.0, soft_floor / 3.0)
                        if debug:
                            if res.facts is None:
                                res.facts = {}
                            res.facts["ambiguous_soft"] = True
                            if res.facts.get("zero_reason"):
                                res.facts["softened_zero_reason"] = res.facts["zero_reason"]
                                res.facts.pop("zero_reason", None)

        positives = [r for r in ranked_results if r.decision == "ACCEPT" and (r.rank_score or 0) > 0]
        zeros = [r for r in ranked_results if r.decision != "ACCEPT" or (r.rank_score or 0) <= 0]

        def _sort_pos(res: Result) -> tuple:
            facts = res.facts or {}
            rank_score = facts.get("final_rank_score", facts.get("rank_score", float("-inf")))
            sim_val = _safe_sim(res.original_corpus_index)
            same_item = facts.get("same_item", None)
            same_item_rank = 2 if same_item is True else (1 if same_item is None else 0)
            same_domain_rank = 1 if facts.get("same_domain") else 0
            return (
                1 if res.decision == "ACCEPT" else 0,
                res.tier if res.tier is not None else -1,
                res.rank_score or 0,
                facts.get("intent_fit", 0.0),
                facts.get("doc_kind") or "",
                sim_val,
                res.original_corpus_index,  # deterministic tie-of-ties
            )

        def _zero_tier(res: Result) -> int:
            return _zero_tier_from_facts(res.facts or {})

        positives.sort(key=_sort_pos, reverse=True)
        zeros.sort(
            key=lambda r: (
                _zero_tier(r),
                -_safe_sim(r.original_corpus_index),
                r.original_corpus_index,
            )
        )
        return positives + zeros
