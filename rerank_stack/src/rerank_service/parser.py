# rerank_stack/src/rerank_service/parser.py
from __future__ import annotations

import json
import re
from typing import Optional, Tuple

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
NUMBERS_RE = re.compile(r"\d+")
FLOAT_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
Labeled_RE = re.compile(r"(?:fine[_ ]?score|FINE_SCORE|score|tie)\s*\"?\s*[:=]\s*(-?\d+(?:\.\d+)?%?)", re.IGNORECASE)
PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")

SINGLE_DIGIT_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}
LEGACY_TO_BUCKET = {
    0: 0,
    3: 1,
    7: 2,
    10: 3,
}


def _bucket(score_float: float) -> int:
    closest = min(LEGACY_TO_BUCKET.keys(), key=lambda b: abs(b - score_float))
    return LEGACY_TO_BUCKET[closest]


def parse_and_bucket(text: str) -> Optional[int]:
    """
    Parses a score from text and maps:
      0/1/2/3 -> 0/1/2/3 (bucket index)
      0/3/7/10 -> 0/1/2/3 (legacy bucket)
    Accepts:
      - single digit 0-3 as a token
      - legacy buckets 0/3/7/10 as a token
      - JSON {"score": 2}
    Returns None when no valid score can be parsed.
    """
    if not text:
        return None

    s = str(text).strip().strip("`").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()

    match = JSON_RE.search(s)
    if match:
        try:
            obj = json.loads(match.group(0))
            val = obj.get("score", obj.get("SCORE", obj.get("Score")))
            if isinstance(val, (int, float)):
                ival = int(val)
                if ival in SINGLE_DIGIT_MAP:
                    return SINGLE_DIGIT_MAP[ival]
                if ival in LEGACY_TO_BUCKET:
                    return LEGACY_TO_BUCKET[ival]
                return _bucket(float(val))
        except Exception:
            pass

    # Find all integers in the string, from right to left.
    numbers = NUMBERS_RE.findall(s)
    for num_str in reversed(numbers):
        score = int(num_str)
        if score in LEGACY_TO_BUCKET:
            return LEGACY_TO_BUCKET[score]
        if score in SINGLE_DIGIT_MAP:
            return SINGLE_DIGIT_MAP[score]

    return None


def parse_fine_score(text: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse a fine_score 0-100 from model output. Returns (score, error_reason).
    """
    if not text:
        return None, "empty_output"
    s = str(text).strip()
    # JSON first
    match = JSON_RE.search(s)
    if match:
        try:
            obj = json.loads(match.group(0))
            val = obj.get("fine_score", obj.get("score", obj.get("tie")))
            if isinstance(val, (int, float)):
                if float(val) <= 1:
                    iv = int(round(float(val) * 100))
                else:
                    iv = int(round(float(val)))
                return max(0, min(100, iv)), None
        except Exception:
            return None, "json_parse_failed"
    # labeled numeric (fine_score / score / tie)
    m = Labeled_RE.search(s)
    if m:
        try:
            token = m.group(1).strip()
            is_percent = token.endswith("%")
            if is_percent:
                token = token[:-1]
            fv = float(token)
            if is_percent:
                iv = int(round(fv))
            elif fv <= 1:
                iv = int(round(fv * 100))
            else:
                iv = int(round(fv))
            return max(0, min(100, iv)), None
        except Exception:
            return None, "parse_int_failed"
    # explicit percent anywhere
    pm = PERCENT_RE.search(s)
    if pm:
        try:
            fv = float(pm.group(1))
            iv = int(round(fv))
            return max(0, min(100, iv)), None
        except Exception:
            pass
    # float or integer anywhere in the text
    floats = FLOAT_RE.findall(s)
    for num_str in reversed(floats):
        try:
            fv = float(num_str)
            iv = int(round(fv if fv > 1 else fv * 100))
            return max(0, min(100, iv)), None
        except Exception:
            continue
    return None, "parse_failed"


def fallback_fine_score(intent_fit: Optional[float], similarity: Optional[float]) -> int:
    """
    Deterministic fallback when tie-break fails: weighted blend of intent_fit and retrieval sim.
    Returns 0-100 int.
    """
    fit = intent_fit if isinstance(intent_fit, (int, float)) else 0.0
    sim = similarity if isinstance(similarity, (int, float)) else 0.0
    score = (0.7 * fit) + (0.3 * sim)
    return max(0, min(100, int(round(score * 100))))
