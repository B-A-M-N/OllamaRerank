# rerank_stack/src/rerank_service/parser.py
from __future__ import annotations

import json
import re
from typing import Optional

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
NUMBERS_RE = re.compile(r'\d+')

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
