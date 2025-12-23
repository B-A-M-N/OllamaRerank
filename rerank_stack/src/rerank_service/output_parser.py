import json
import re
from typing import Optional, Tuple

# Regex to find the first standalone 0, 3, 7, or 10 in the text.
BUCKET_RE = re.compile(r"\b(0|3|7|10)\b")
SCORE_TAG_RE = re.compile(r"\bscore\b\s*[:=]\s*(10|7|3|0)\b", re.IGNORECASE)

def extract_score_bucket_with_reason(raw: str) -> Tuple[Optional[int], str]:
    """
    Extracts a valid bucket score {0, 3, 7, 10} from raw model output.
    Returns (score, reason) where score is None if parsing failed.
    """
    if not raw:
        return None, "empty_output"

    text = raw.strip().strip("`").strip()
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        text = text[1:-1].strip()

    # 1) JSON-ish parse (first {...} block)
    if "{" in text and "}" in text:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            blob = match.group(0)
            try:
                obj = json.loads(blob)
                val = obj.get("score", obj.get("SCORE", obj.get("Score")))
                if isinstance(val, (int, float)) and int(val) in (0, 3, 7, 10):
                    return int(val), "json"
            except Exception:
                pass

    # 2) Labeled formats
    match = SCORE_TAG_RE.search(text)
    if match:
        return int(match.group(1)), "score_tag"

    # 3) Bare allowed token anywhere
    match = BUCKET_RE.search(text)
    if match:
        return int(match.group(1)), "bare_token"

    return None, "no_match"


def extract_score_bucket(raw: str) -> Optional[int]:
    """
    Extracts a valid bucket score {0, 3, 7, 10} from raw model output.
    Accepts JSON, labeled formats (score: 7), or a bare digit anywhere.
    """
    score, _ = extract_score_bucket_with_reason(raw)
    return score
