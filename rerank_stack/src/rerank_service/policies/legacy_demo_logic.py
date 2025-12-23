from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from ..core.quality import QualitySignals, analyze_signals
except Exception:  # pragma: no cover - fallback for alternate import paths
    from rerank_service.core.quality import QualitySignals, analyze_signals  # type: ignore

from ..policy import PolicyContext


# -------------------------
# Legacy/demo heuristics
# -------------------------

ACTION_VERBS = {
    "turn", "shut", "close", "open", "remove", "unscrew", "screw", "tighten", "loosen",
    "pull", "push", "lift", "press", "install", "replace", "swap", "change", "inspect",
    "check", "test", "clean", "lube", "lubricate", "drain", "flush", "bleed", "fix",
    "disconnect", "reconnect", "reassemble", "assemble", "align", "torque",
    "seal", "apply", "use", "measure",
}

STEP_MARKERS = {
    "step", "steps", "1)", "2)", "3)", "first", "second", "then", "next", "finally",
    "before", "after", "start by", "finish by",
}

PART_WORDS = {
    # plumbing
    "cartridge", "handle", "valve", "washer", "o-ring", "oring", "aerator", "stem",
    "packing", "nut", "seat", "spout", "shower", "faucet", "pipe", "supply", "drain",
    # automotive
    "spark", "plug", "coil", "socket", "brake", "pads", "rotor", "oil", "filter", "torque",
    # bicycle
    "chain", "tire", "tires", "tube", "cassette", "derailleur",
    # generic repair/maintenance
    "belt", "hose",
}

EXPLANATORY_HINTS = {
    "understanding", "guide to", "what is", "types of", "symptoms", "overview", "introduction"
}

_WORD_RE = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _contains_step_marker(text_l: str) -> bool:
    if re.search(r"(^|\s)\d+[\.\)]\s", text_l):
        return True
    return any(m in text_l for m in STEP_MARKERS)


def _count_parts(tokens: List[str]) -> int:
    parts = 0
    for t in tokens:
        if t in PART_WORDS:
            parts += 1
        elif t.endswith("s") and t[:-1] in PART_WORDS:
            parts += 1
    return parts


def _count_punct_signals(text: str) -> int:
    return text.count(",") + text.count(";")


def _explanatory_score(text_l: str) -> int:
    return sum(1 for h in EXPLANATORY_HINTS if h in text_l)


@dataclass
class LegacyDocQuality:
    procedural: bool
    action_verbs: int
    part_terms: int
    has_step_markers: bool
    length_chars: int
    length_tokens: int
    punct_signals: int
    explanatory_hints: int
    quality_score: float


def analyze_doc(text: str) -> LegacyDocQuality:
    raw = text or ""
    text_l = raw.lower().strip()
    tokens = _tokenize(text_l)
    signals: QualitySignals = analyze_signals(text)

    part_terms = _count_parts(tokens)
    has_step_markers = _contains_step_marker(text_l)
    punct_signals = _count_punct_signals(raw)
    explanatory_hints = _explanatory_score(text_l)

    procedural = (
        has_step_markers
        or signals.action_verbs >= 2
        or (signals.action_verbs >= 1 and part_terms >= 1)
    )

    score = 0.0
    score += 0.25 * min(signals.action_verbs, 4) / 4.0
    score += 0.25 * min(part_terms, 4) / 4.0
    score += 0.20 if has_step_markers else 0.0
    score += 0.20 * min(signals.length_tokens, 40) / 40.0
    score += 0.10 * min(punct_signals, 6) / 6.0
    score -= 0.10 * min(explanatory_hints, 3) / 3.0
    score = max(0.0, min(1.0, score))

    return LegacyDocQuality(
        procedural=procedural,
        action_verbs=signals.action_verbs,
        part_terms=part_terms,
        has_step_markers=has_step_markers,
        length_chars=signals.length_chars,
        length_tokens=signals.length_tokens,
        punct_signals=punct_signals,
        explanatory_hints=explanatory_hints,
        quality_score=score,
    )


HOWTO_TRIGGERS = {
    "how", "fix", "replace", "repair", "install", "remove", "troubleshoot",
    "steps", "guide", "tutorial", "change",
}


def is_howto_query(query: str) -> bool:
    q = (query or "").lower()
    toks = set(_tokenize(q))
    return any(t in HOWTO_TRIGGERS for t in toks)


STOP_WORDS = {
    "how", "to", "a", "the", "and", "or", "of", "in", "for", "on",
    "is", "are", "was", "were", "be", "it", "this", "that", "with",
}


def _keyword_overlap(query: str, doc: str) -> int:
    q_tokens = {t for t in _tokenize(query) if len(t) >= 3 and t not in STOP_WORDS}
    d_tokens = {t for t in _tokenize(doc) if len(t) >= 3 and t not in STOP_WORDS}
    return len(q_tokens & d_tokens)


def _is_ambiguous_query(query: str, context: Optional[PolicyContext]) -> bool:
    if not context or not context.ambiguous_triggers:
        return False
    toks = set(_tokenize(query))
    if not toks:
        return True
    if len(toks) <= 1:
        return True
    return bool(toks & context.ambiguous_triggers)


INTENT_HOWTO = {"how", "fix", "replace", "install", "remove", "repair", "troubleshoot", "steps", "guide", "tutorial", "change"}
INTENT_OVERVIEW = {"maintenance", "maintaining", "maintain", "basics", "guide", "overview", "tips", "schedule", "routine", "service", "interval", "checklist"}
INTENT_DIAGNOSE = {"symptoms", "why", "cause", "diagnose", "problem", "issues"}
INTENT_INFO = {"types", "difference", "understanding", "what", "explain"}


def detect_query_intent(query: str) -> str:
    toks = set(_tokenize(query))
    if toks & INTENT_HOWTO:
        return "HOWTO"
    if toks & INTENT_DIAGNOSE:
        return "DIAGNOSE"
    if toks & INTENT_OVERVIEW:
        return "OVERVIEW"
    if toks & INTENT_INFO:
        return "INFO"
    return "UNKNOWN"


def detect_doc_kind(doc: str, quality: LegacyDocQuality) -> str:
    doc_l = (doc or "").lower()
    toks = set(_tokenize(doc_l))

    maint_tokens = {"maintenance", "maintaining", "maintain", "service", "servicing", "interval", "schedule", "routine", "checklist"}
    procedural_triggers = {"how to", "steps to", "step-by-step", "step by step"}
    oil_hits = any(
        phrase in doc_l
        for phrase in [
            "engine oil",
            "oil change",
            "oil filter",
            "viscosity",
            "synthetic",
            "miles",
            "interval",
        ]
    )

    if any(trigger in doc_l for trigger in procedural_triggers):
        return "PROCEDURAL"

    if any(tok in toks for tok in {"fix", "repair", "replace", "install", "remove", "leak", "drip"}):
        if quality.action_verbs >= 1:
            return "PROCEDURAL"

    maint_hits = bool(toks & maint_tokens)
    if maint_hits or oil_hits:
        if quality.action_verbs >= 2 or quality.has_step_markers:
            return "PROCEDURAL"
        return "OVERVIEW"

    if any(phrase in doc_l for phrase in ["types of", "understanding", "overview", "guide to"]):
        return "INFORMATIONAL"
    if toks & INTENT_DIAGNOSE or "symptoms" in doc_l or "signs of" in doc_l:
        return "DIAGNOSTIC"
    if toks & INTENT_OVERVIEW:
        return "OVERVIEW"
    return "UNKNOWN"


def intent_fit_score(query_intent: str, doc_kind: str) -> float:
    # Broad maintenance intent: boost OVERVIEW/INFO when query is broad.
    broad_intent_tokens = {"maintenance", "service", "upkeep", "care", "routine", "schedule"}
    matrix = {
        "HOWTO": {
            "PROCEDURAL": 1.0,
            "DIAGNOSTIC": 0.7,
            "OVERVIEW": 0.6,
            "INFORMATIONAL": 0.4,
            "UNKNOWN": 0.5,
        },
        "OVERVIEW": {
            "OVERVIEW": 1.0,
            "PROCEDURAL": 0.75,
            "DIAGNOSTIC": 0.7,
            "INFORMATIONAL": 0.45,
            "UNKNOWN": 0.6,
        },
        "DIAGNOSE": {
            "DIAGNOSTIC": 1.0,
            "PROCEDURAL": 0.7,
            "OVERVIEW": 0.6,
            "INFORMATIONAL": 0.5,
            "UNKNOWN": 0.5,
        },
        "INFO": {
            "INFORMATIONAL": 1.0,
            "OVERVIEW": 0.7,
            "DIAGNOSTIC": 0.6,
            "PROCEDURAL": 0.5,
            "UNKNOWN": 0.5,
        },
        "UNKNOWN": {
            "PROCEDURAL": 0.85,
            "DIAGNOSTIC": 0.85,
            "OVERVIEW": 0.85,
            "INFORMATIONAL": 0.85,
            "UNKNOWN": 0.85,
        },
    }
    base = matrix.get(query_intent, matrix["UNKNOWN"]).get(doc_kind, 0.5)
    if query_intent in {"OVERVIEW", "INFO", "UNKNOWN"} and doc_kind in {"OVERVIEW", "INFORMATIONAL"}:
        base = min(1.0, base + 0.15)
    return base


def model_score_is_flat(scores: List[float]) -> bool:
    uniq = {round(s, 6) for s in scores}
    return len(uniq) <= 2


def rerank_postprocess(
    query: str,
    docs: List[str],
    rerank_scores: List[float],
    original_similarities: Optional[List[float]] = None,
    tie_breakers: Optional[List[float]] = None,
    context: Optional[PolicyContext] = None,
) -> List[Tuple[int, float, float, LegacyDocQuality, str, str, float]]:
    """
    Returns a sorted list of (doc_index, gate_score, final_rank_score, quality, query_intent, doc_kind, intent_fit).
    gate_score = coarse bucket from the model/rules (0/1/2/3).
    final_rank_score = intent-aware refinement used only for ordering among accepted docs.
    """
    assert len(docs) == len(rerank_scores)
    if original_similarities is not None:
        assert len(original_similarities) == len(docs)
    if tie_breakers is not None:
        assert len(tie_breakers) == len(docs)

    query_intent = detect_query_intent(query)

    def _base_action_score(doc_text: str, quality: LegacyDocQuality) -> float:
        doc_toks = set(_tokenize(doc_text))
        has_schedule = bool(doc_toks & {"maintenance", "service", "schedule", "interval", "routine", "inspect"})
        model_specific = bool(re.search(r"\b(19|20)\d{2}\b", doc_text))
        part_bonus = 0.1 if quality.part_terms > 0 else 0.0
        specificity_penalty = 0.1 if model_specific else 0.0

        procedural_hits = (
            quality.action_verbs
            + (2 if quality.has_step_markers else 0)
            + min(quality.punct_signals, 3)
        )

        score = 0.5
        if procedural_hits >= 5:
            score += 0.2
        elif procedural_hits >= 3:
            score += 0.1
        if has_schedule:
            score += 0.1

        score += part_bonus
        score -= specificity_penalty
        score = max(0.0, min(1.0, score))
        return score

    ambiguous = _is_ambiguous_query(query, context)
    short_query = len(_tokenize(query)) <= 1
    scored = []
    for i, (doc, s) in enumerate(zip(docs, rerank_scores)):
        q = analyze_doc(doc)
        sim = original_similarities[i] if original_similarities else 0.0
        tb = tie_breakers[i] if tie_breakers else 0.0
        final = float(s)
        overlap = _keyword_overlap(query, doc)
        doc_kind = detect_doc_kind(doc, q)
        action_score = _base_action_score(doc, q)
        intent_fit = intent_fit_score(query_intent, doc_kind)
        final_rank = action_score * intent_fit if final > 0 else 0.0
        scored.append((i, final, sim, q, overlap, tb, action_score, query_intent, doc_kind, intent_fit, final_rank))

    if (ambiguous or short_query) and model_score_is_flat(rerank_scores) and original_similarities is not None:
        scored = [
            (i, (0.85 * sim) + (0.15 * overlap), sim, q, overlap, tb, action_score, q_intent, d_kind, intent_fit, final_rank)
            for (i, _s, sim, q, overlap, tb, action_score, q_intent, d_kind, intent_fit, final_rank) in scored
        ]
        scored.sort(key=lambda item: (-item[1], -item[4], -item[3].quality_score, item[0]))
    else:
        pos = [row for row in scored if row[1] > 0]
        neg = [row for row in scored if row[1] <= 0]

        pos.sort(
            key=lambda item: (
                -item[1],
                -item[10],
                -item[5],
                -item[4],
                -item[3].quality_score,
                -item[2],
                item[0],
            )
        )

        neg.sort(
            key=lambda item: (
                -item[1],
                -item[5],
                -item[4],
                -item[3].quality_score,
                -item[2],
                item[0],
            )
        )

        scored = pos + neg

    return [
        (
            item[0],
            item[1],
            item[10] if item[1] > 0 else 0.0,
            item[3],
            item[7],
            item[8],
            item[9],
        )
        for item in scored
    ]
