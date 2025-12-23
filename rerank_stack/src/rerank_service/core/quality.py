from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# -------------------------
# Generic heuristic signals
# -------------------------

GENERIC_ACTION_VERBS = {
    "turn", "shut", "close", "open", "remove", "unscrew", "screw", "tighten", "loosen",
    "pull", "push", "lift", "press", "install", "replace", "swap", "change", "inspect",
    "check", "test", "clean", "drain", "flush", "bleed", "disconnect", "reconnect",
    "reassemble", "assemble", "align", "torque", "seal", "apply", "use", "measure",
    "fix", "repair", "lube", "lubricate", "service", "maintain",
}

STEP_MARKERS = {
    "step", "steps", "1)", "2)", "3)", "first", "second", "then", "next", "finally",
    "before", "after", "start by", "finish by",
}

EXPLANATORY_HINTS = {
    "understanding", "guide to", "what is", "types of", "symptoms", "overview", "introduction",
}

_WORD_RE = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _contains_step_marker(text_l: str) -> bool:
    if re.search(r"(^|\s)\d+[\.\)]\s", text_l):
        return True
    return any(m in text_l for m in STEP_MARKERS)


def _count_action_verbs(tokens: List[str]) -> int:
    return sum(1 for t in tokens if t in GENERIC_ACTION_VERBS)


def _count_punct_signals(text: str) -> int:
    return text.count(",") + text.count(";")


def _explanatory_score(text_l: str) -> int:
    return sum(1 for h in EXPLANATORY_HINTS if h in text_l)


@dataclass
class QualitySignals:
    action_verbs: int
    has_step_markers: bool
    length_chars: int
    length_tokens: int
    punct_signals: int
    explanatory_hints: int


def analyze_signals(text: str) -> QualitySignals:
    raw = text or ""
    text_l = raw.lower().strip()
    tokens = _tokenize(text_l)

    action_verbs = _count_action_verbs(tokens)
    has_step_markers = _contains_step_marker(text_l)
    length_chars = len(raw)
    length_tokens = len(tokens)
    punct_signals = _count_punct_signals(raw)
    explanatory_hints = _explanatory_score(text_l)

    return QualitySignals(
        action_verbs=action_verbs,
        has_step_markers=has_step_markers,
        length_chars=length_chars,
        length_tokens=length_tokens,
        punct_signals=punct_signals,
        explanatory_hints=explanatory_hints,
    )


def annotate_corpus(docs: List[str]) -> List[dict]:
    return [analyze_signals(d).__dict__.copy() for d in docs]
