"""
Generic policy placeholder.

This policy does no domain/item gating and only uses generic quality/intent
signals. It is meant as the "all systems" baseline; the demo should
explicitly load the legacy demo policy to preserve its behavior.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rerank_service.policies import BasePolicy
from rerank_service.schemas import Document, Result, Options
from rerank_service.policy import Policy, build_policy_context, PolicyContext
import re
from ..core.quality import analyze_signals

_WORD_RE = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

def _tokenize(text: str) -> set[str]:
    return set(t.lower() for t in _WORD_RE.findall(text or ""))

INTENT_HOWTO = {"how", "fix", "replace", "install", "remove", "repair", "troubleshoot", "steps", "guide", "tutorial", "change"}
INTENT_OVERVIEW = {"maintenance", "maintaining", "maintain", "basics", "guide", "overview", "tips", "schedule", "routine", "service", "interval", "checklist"}
INTENT_DIAGNOSE = {"symptoms", "why", "cause", "diagnose", "problem", "issues"}
INTENT_INFO = {"types", "difference", "understanding", "what", "explain"}
DOMAIN_PLUMBING = {"shower", "faucet", "plumbing", "cartridge", "valve", "pipe", "water", "heater", "drain"}
DOMAIN_AUTO = {"car", "auto", "automotive", "engine", "brake", "oil", "spark", "torque", "lug"}
DOMAIN_BIKE = {"bicycle", "bike", "chain", "cassette", "derailleur", "tire", "wheel"}

def detect_query_intent(query: str) -> str:
    toks = _tokenize(query)
    if toks & INTENT_HOWTO:
        return "HOWTO"
    if toks & INTENT_DIAGNOSE:
        return "DIAGNOSE"
    if toks & INTENT_OVERVIEW:
        return "OVERVIEW"
    if toks & INTENT_INFO:
        return "INFO"
    return "UNKNOWN"

def detect_doc_kind(doc: str, signals) -> str:
    doc_l = (doc or "").lower()
    toks = _tokenize(doc_l)
    if any(phrase in doc_l for phrase in ["types of", "understanding", "overview", "guide to"]):
        return "INFORMATIONAL"
    if "symptoms" in doc_l or "signs of" in doc_l:
        return "DIAGNOSTIC"
    if signals.action_verbs >= 2 or signals.has_step_markers:
        return "PROCEDURAL"
    if toks & INTENT_OVERVIEW:
        return "OVERVIEW"
    return "UNKNOWN"

def intent_fit_score(query_intent: str, doc_kind: str) -> float:
    matrix = {
        "HOWTO": {"PROCEDURAL": 1.0, "DIAGNOSTIC": 0.7, "OVERVIEW": 0.6, "INFORMATIONAL": 0.4, "UNKNOWN": 0.5},
        "OVERVIEW": {"OVERVIEW": 1.0, "PROCEDURAL": 0.75, "DIAGNOSTIC": 0.7, "INFORMATIONAL": 0.45, "UNKNOWN": 0.6},
        "DIAGNOSE": {"DIAGNOSTIC": 1.0, "PROCEDURAL": 0.7, "OVERVIEW": 0.6, "INFORMATIONAL": 0.5, "UNKNOWN": 0.5},
        "INFO": {"INFORMATIONAL": 1.0, "OVERVIEW": 0.7, "DIAGNOSTIC": 0.6, "PROCEDURAL": 0.5, "UNKNOWN": 0.5},
        "UNKNOWN": {"PROCEDURAL": 0.85, "DIAGNOSTIC": 0.85, "OVERVIEW": 0.85, "INFORMATIONAL": 0.85, "UNKNOWN": 0.85},
    }
    return matrix.get(query_intent, matrix["UNKNOWN"]).get(doc_kind, 0.5)


def detect_domain(text: str) -> str:
    toks = _tokenize(text)
    scores = {
        "plumbing": len(toks & DOMAIN_PLUMBING),
        "automotive": len(toks & DOMAIN_AUTO),
        "bicycle": len(toks & DOMAIN_BIKE),
    }
    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] == 0:
        return "unknown"
    winners = [k for k, v in scores.items() if v == best[1]]
    if len(winners) > 1:
        return "ambiguous"
    return winners[0]


def is_ambiguous_query(query: str) -> bool:
    toks = _tokenize(query)
    if not toks or len(toks) <= 1:
        return True
    domain = detect_domain(query)
    return domain in {"unknown", "ambiguous"}


class GenericPolicy(BasePolicy):
    def policy_id(self) -> str:
        return "generic"

    def preprocess(self, query: str) -> dict:
        q_domain = detect_domain(query)
        mode = "AMBIGUOUS" if is_ambiguous_query(query) else "SPECIFIC"
        return {"query_intent": detect_query_intent(query), "query_domain": q_domain, "mode": mode}

    def doc_features(self, query: str, doc: str, ctx: dict) -> dict:
        quality = analyze_signals(doc)
        doc_kind = detect_doc_kind(doc, quality)
        q_intent = ctx.get("query_intent", "UNKNOWN")
        fit = intent_fit_score(q_intent, doc_kind)
        doc_domain = detect_domain(doc)
        query_domain = ctx.get("query_domain", "unknown")
        domain_penalty = 0.0
        if query_domain not in {"unknown", "ambiguous"} and doc_domain not in {"unknown", "ambiguous"}:
            if query_domain != doc_domain:
                domain_penalty = 0.25
        # generic policy does no domain/item gating; just returns signals
        return {
            "quality": quality,
            "doc_kind": doc_kind,
            "query_intent": q_intent,
            "intent_fit": fit,
            "doc_domain": doc_domain,
            "query_domain": query_domain,
            "domain_penalty": domain_penalty,
        }

    def build_context(self, query: str, policy_id_override: Optional[str] = None) -> PolicyContext:
        # Build a minimal empty PolicyContext to satisfy the scorer (no domain gating).
        policy = Policy(name="generic", scale=[0, 1, 2, 3])
        return build_policy_context(policy)

    def precompute_facts(self, query: str, documents: List[Document]) -> Dict[int, dict]:
        return {doc.original_index: {} for doc in documents}

    def clamp_bucket(self, score: int, facts: dict) -> int:
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
        # Generic: no gating, use model score + similarity + lightweight intent/doc kind.
        ctx = self.preprocess(query)
        mode = ctx.get("mode", "SPECIFIC")
        results: List[Tuple[Result, tuple]] = []
        sim_by_req = {
            i: similarities_by_req[i] if i < len(similarities_by_req) else None
            for i in range(len(documents))
        }
        doc_by_orig = {doc.original_index: doc for doc in documents}

        for req_idx, orig_idx, score, _tb in scored_docs:
            doc_obj = doc_by_orig.get(orig_idx)
            doc_text = doc_obj.text if doc_obj else ""
            sim = sim_by_req.get(req_idx) or 0.0
            feats = self.doc_features(query, doc_text, ctx)
            intent_fit = feats.get("intent_fit", 0.5)
            quality = feats.get("quality")
            action_score = quality.action_verbs / 4.0 if quality else 0.0
            base_pre_rank_score = max(0.0, min(1.0, action_score * intent_fit))
            pre_rank_score = base_pre_rank_score
            if feats.get("domain_penalty"):
                pre_rank_score = max(0.0, pre_rank_score - feats["domain_penalty"])
            score_raw = float(score)
            score_norm = max(0.0, min(1.0, score_raw / 3.0))
            tier = int(score_raw)

            decision = "ACCEPT"
            zero_reason = None
            rank_score_reason = "accepted"
            rank_score = pre_rank_score
            decision_priority = 2  # ACCEPT highest

            if mode == "SPECIFIC":
                qd = feats.get("query_domain")
                dd = feats.get("doc_domain")
                if qd not in {"unknown", "ambiguous"} and dd not in {"unknown", "ambiguous"} and qd != dd:
                    decision = "REJECT_TIER_9"
                    zero_reason = "domain_mismatch"
                elif tier <= 0:
                    decision = "REJECT_TIER_2"
                    zero_reason = zero_reason or "low_signal"
            else:  # AMBIGUOUS
                if tier <= 0:
                    # Keep low-signal candidates but heavily downweight them.
                    decision = "KEEP_LOW_SIGNAL"
                    decision_priority = 1
                    rank_score_reason = "accepted_low_signal"
                    rank_score = max(0.0, pre_rank_score * 0.35)
                    zero_reason = "low_signal"

            if decision != "ACCEPT":
                rank_score = 0.0
                rank_score_reason = "rejected"
                if pre_rank_score > 0.5:
                    print(
                        "[generic warn] High pre_rank_score but rejected",
                        {"orig_idx": orig_idx, "pre_rank_score": pre_rank_score, "decision": decision, "zero_reason": zero_reason},
                    )

            doc_kind = feats.get("doc_kind") or "UNKNOWN"
            doc_kind_rank = {
                "PROCEDURAL": 4,
                "DIAGNOSTIC": 3,
                "OVERVIEW": 2,
                "INFORMATIONAL": 1,
                "UNKNOWN": 0,
            }.get(str(doc_kind).upper(), 0)

            sort_key = (
                decision_priority,  # ACCEPT > KEEP_LOW_SIGNAL > rejects
                score_raw,          # coarse bucket
                rank_score,         # continuous ordering
                intent_fit,         # intent alignment
                doc_kind_rank,      # prefer procedural when relevant
                sim,                # retrieval tie-break
            )

            facts: Dict[str, object] = {
                "query_intent": feats.get("query_intent"),
                "doc_kind": feats.get("doc_kind"),
                "intent_fit": intent_fit,
                "final_rank_score": pre_rank_score,
                "rank_score": rank_score,
                "policy_id": "generic",
                "tier": tier,
                "score_raw": score_raw,
                "score_norm": score_norm,
                "decision": decision,
                "mode": mode,
                "pre_rank_score": pre_rank_score,
                "pre_rank_score_raw": base_pre_rank_score,
                "rank_score_reason": rank_score_reason,
                "doc_domain": feats.get("doc_domain"),
                "query_domain": feats.get("query_domain"),
            }
            if zero_reason:
                facts["zero_reason"] = zero_reason
            if not debug:
                # Strip to core when not in debug mode but keep non-null dict
                facts = {
                    "policy_id": "generic",
                    "tier": tier,
                    "score_raw": score_raw,
                    "score_norm": score_norm,
                    "decision": decision,
                    "mode": mode,
                    "pre_rank_score": pre_rank_score,
                    "pre_rank_score_raw": base_pre_rank_score,
                    "rank_score": rank_score,
                    "rank_score_reason": rank_score_reason,
                    "doc_domain": feats.get("doc_domain"),
                    "query_domain": feats.get("query_domain"),
                    "doc_kind": feats.get("doc_kind"),
                    "query_intent": feats.get("query_intent"),
                }
                if zero_reason:
                    facts["zero_reason"] = zero_reason

            results.append(
                (
                    Result(
                        index=0,
                        original_corpus_index=orig_idx,
                        score=score_raw,
                        pre_rank_score=pre_rank_score,
                        rank_score=rank_score,
                        rank_score_reason=rank_score_reason,
                        score_raw=score_raw,
                        score_norm=score_norm,
                        tier=tier,
                        decision=decision,
                        policy_id="generic",
                        facts=facts,
                    ),
                    sort_key,
                )
            )

        results.sort(key=lambda item: item[1], reverse=True)
        ordered: List[Result] = []
        accepts = [pair for pair in results if pair[0].decision == "ACCEPT"]
        keeps = [pair for pair in results if pair[0].decision == "KEEP_LOW_SIGNAL"]
        rejects = [pair for pair in results if pair[0].decision not in {"ACCEPT", "KEEP_LOW_SIGNAL"}]

        if mode == "AMBIGUOUS" and accepts:
            by_domain: Dict[str, List[Tuple[Result, float]]] = {}
            for res, ss in accepts:
                dom = res.facts.get("doc_domain") if isinstance(res.facts, dict) else "unknown"
                by_domain.setdefault(dom or "unknown", []).append((res, ss))
            diversified: List[Tuple[Result, float]] = []
            for dom, bucket in by_domain.items():
                bucket.sort(key=lambda item: item[0].rank_score or 0, reverse=True)
                diversified.append(bucket[0])
            # Fill remaining slots by global rank_score
            remaining = [
                pair for pair in accepts if pair not in diversified
            ]
            remaining.sort(key=lambda item: item[0].rank_score or 0, reverse=True)
            accepts = diversified + remaining

        accepts.sort(key=lambda item: item[1], reverse=True)
        keeps.sort(key=lambda item: item[1], reverse=True)
        rejects.sort(key=lambda item: item[1], reverse=True)

        ordered_pairs = accepts + keeps + rejects
        for i, (res, _) in enumerate(ordered_pairs):
            res.index = i
            ordered.append(res)
        return ordered
