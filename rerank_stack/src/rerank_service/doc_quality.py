# rerank_stack/src/rerank_service/doc_quality.py
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from .policy import PolicyContext


# -------------------------
# Heuristic dictionaries
# -------------------------

# Verbs that usually indicate concrete actions / steps.
ACTION_VERBS = {
    "turn", "shut", "close", "open", "remove", "unscrew", "screw", "tighten", "loosen",
    "pull", "push", "lift", "press", "install", "replace", "swap", "change", "inspect",
    "check", "test", "clean", "lube", "lubricate", "drain", "flush", "bleed",
    "disconnect", "reconnect", "reassemble", "assemble", "align", "torque",
    "seal", "apply", "use", "measure",
}

# “Step markers” are strong signals of procedure even if verbs are missing.
STEP_MARKERS = {
    "step", "steps", "1)", "2)", "3)", "first", "second", "then", "next", "finally",
    "before", "after", "start by", "finish by",
}

# Parts / objects: you can grow this per domain.
# (Not required, but helps distinguish “turn off water” vs random advice.)
PART_WORDS = {
    # plumbing
    "cartridge", "handle", "valve", "washer", "o-ring", "oring", "aerator", "stem",
    "packing", "nut", "seat", "spout", "shower", "faucet", "pipe", "supply", "drain",
    # automotive
    "spark", "plug", "coil", "socket", "brake", "pads", "rotor", "oil", "filter", "torque",
    # bicycle
    "chain", "tire", "tires", "tube", "cassette", "derailleur",
}

# “Non-procedure” hints: text that is usually explanatory not actionable
# (we don’t *penalize* too hard; this is just a signal).
EXPLANATORY_HINTS = {
    "understanding", "guide to", "what is", "types of", "symptoms", "overview", "introduction"
}


# -------------------------
# Helpers
# -------------------------

_WORD_RE = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]

def _contains_step_marker(text_l: str) -> bool:
    # quick checks for enumerations like "1." "2." or "1)" "2)"
    if re.search(r"(^|\s)\d+[\.\)]\s", text_l):
        return True
    return any(m in text_l for m in STEP_MARKERS)

def _count_action_verbs(tokens: List[str]) -> int:
    return sum(1 for t in tokens if t in ACTION_VERBS)

def _count_parts(tokens: List[str]) -> int:
    # allow o-ring variants and simple plural handling
    parts = 0
    for t in tokens:
        if t in PART_WORDS:
            parts += 1
        elif t.endswith("s") and t[:-1] in PART_WORDS:  # crude plural match
            parts += 1
    return parts

def _count_punct_signals(text: str) -> int:
    # commas/semicolons often correlate with step lists in your corpus style
    return text.count(",") + text.count(";")

def _explanatory_score(text_l: str) -> int:
    return sum(1 for h in EXPLANATORY_HINTS if h in text_l)


# -------------------------
# Output type
# -------------------------

@dataclass
class DocQuality:
    procedural: bool
    action_verbs: int
    part_terms: int
    has_step_markers: bool
    length_chars: int
    length_tokens: int
    punct_signals: int
    explanatory_hints: int
    quality_score: float  # 0..1-ish, heuristic


def analyze_doc(text: str) -> DocQuality:
    raw = text or ""
    text_l = raw.lower().strip()
    tokens = _tokenize(text_l)

    action_verbs = _count_action_verbs(tokens)
    part_terms = _count_parts(tokens)
    has_step_markers = _contains_step_marker(text_l)
    length_chars = len(raw)
    length_tokens = len(tokens)
    punct_signals = _count_punct_signals(raw)
    explanatory_hints = _explanatory_score(text_l)

    # --- Procedural decision ---
    # We call it "procedural" if:
    # - step markers exist OR
    # - there are >=2 action verbs OR
    # - there is >=1 action verb + >=1 part term (action on an object)
    procedural = (
        has_step_markers
        or action_verbs >= 2
        or (action_verbs >= 1 and part_terms >= 1)
    )

    # --- Quality score (for tie-breaking, gating, etc.) ---
    # Score is mostly driven by action + part specificity, lightly by length and step markers.
    # Explanatory hints nudge down slightly (not a punishment, just a nudge).
    score = 0.0
    score += 0.25 * min(action_verbs, 4) / 4.0
    score += 0.25 * min(part_terms, 4) / 4.0
    score += 0.20 if has_step_markers else 0.0
    score += 0.20 * min(length_tokens, 40) / 40.0
    score += 0.10 * min(punct_signals, 6) / 6.0
    score -= 0.10 * min(explanatory_hints, 3) / 3.0
    score = max(0.0, min(1.0, score))

    return DocQuality(
        procedural=procedural,
        action_verbs=action_verbs,
        part_terms=part_terms,
        has_step_markers=has_step_markers,
        length_chars=length_chars,
        length_tokens=length_tokens,
        punct_signals=punct_signals,
        explanatory_hints=explanatory_hints,
        quality_score=score,
    )


def annotate_corpus(docs: List[str]) -> List[Dict]:
    return [asdict(analyze_doc(d)) for d in docs]


# -------------------------
# Optional: query intent detection
# -------------------------

HOWTO_TRIGGERS = {
    "how", "fix", "replace", "repair", "install", "remove", "troubleshoot",
    "steps", "guide", "tutorial", "change",
}

def is_howto_query(query: str) -> bool:
    q = (query or "").lower()
    toks = set(_tokenize(q))
    # “how to” gets caught by "how"
    return any(t in HOWTO_TRIGGERS for t in toks)


# -------------------------
# Optional: post-rerank tie-break & gating
# -------------------------
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

def model_score_is_flat(scores: List[float]) -> bool:
    uniq = {round(s, 6) for s in scores}
    return len(uniq) <= 2


def rerank_postprocess(
    query: str,
    docs: List[str],
    rerank_scores: List[float],
    original_similarities: Optional[List[float]] = None,
    context: Optional[PolicyContext] = None,
) -> List[Tuple[int, float, DocQuality]]:
    """
    Returns a sorted list of (doc_index, final_score, quality).
    """
    assert len(docs) == len(rerank_scores)
    if original_similarities is not None:
        assert len(original_similarities) == len(docs)

    ambiguous = _is_ambiguous_query(query, context)
    short_query = len(_tokenize(query)) <= 1
    scored = []
    for i, (doc, s) in enumerate(zip(docs, rerank_scores)):
        q = analyze_doc(doc)
        sim = original_similarities[i] if original_similarities else 0.0
        final = float(s)
        overlap = _keyword_overlap(query, doc)
        scored.append((i, final, sim, q, overlap))

    # If scores are flat AND query is ambiguous/short AND we have similarity, use similarity+overlap as the score (skip model signal).
    if (ambiguous or short_query) and model_score_is_flat(rerank_scores) and original_similarities is not None:
        scored = [
            (i, (0.85 * sim) + (0.15 * overlap), sim, q, overlap)
            for (i, _s, sim, q, overlap) in scored
        ]
        scored.sort(key=lambda item: (-item[1], -item[4], -item[3].quality_score, item[0]))
    else:
        # Sort by: 1. rerank_score desc, 2. query/doc lexical overlap score desc,
        # 3. procedural-ness score desc, 4. initial_similarity desc, 5. original_idx asc
        scored.sort(key=lambda item: (-item[1], -item[4], -item[3].quality_score, -item[2], item[0]))

    # Reconstruct the final list of tuples
    return [(item[0], item[1], item[3]) for item in scored]
