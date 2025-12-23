from __future__ import annotations
import asyncio
import httpx
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set

from .ollama_client import OllamaClient
from .policy import PolicyContext
from .schemas import Options
from .parser import parse_and_bucket
from .prompt_builder import build_rerank_prompt_single_doc, build_retry_score_prompt

SEM = asyncio.Semaphore(1)


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "how",
    "what", "why", "is", "are", "was", "were", "be", "it", "this", "that",
    "i", "you", "we", "they"
}

_word_re = re.compile(r"[a-z0-9]+")
STEP_BULLET_RE = re.compile(r'(?m)^\s*(?:\d+[\)\.\:]|[-*+])\s+')
STEP_SEPARATOR_RE = re.compile(r'[.;:]\s+|,\s+')
VERB_RE = re.compile(
    r'\b(turn|remove|install|replace|tighten|test|open|close|pull|push|unscrew|screw|disconnect|reconnect|shut|start|stop|run|check|clean|reassemble|drain)\b',
    re.I,
)
STEP_WORDS_RE = re.compile(r'\b(step|first|then|next|after|finally|before|afterward|once)\b', re.I)


def document_step_summary(text: str) -> Tuple[int, int, bool, bool]:
    bullets = len(STEP_BULLET_RE.findall(text or ""))
    verbs = len(set(m.group(1).lower() for m in VERB_RE.finditer(text or "")))
    step_word_flag = bool(STEP_WORDS_RE.search(text or ""))
    text_low = (text or "").lower()
    has_water_off = "turn off" in text_low or "shut off" in text_low
    return bullets, verbs, step_word_flag, has_water_off


def normalize(word: str) -> str:
    w = word.lower()
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w

def tokenize(text: str) -> list[str]:
    return [w for w in _word_re.findall((text or "").lower()) if w not in _STOPWORDS and len(w) >= 3]

def canonicalize(word: str, synonym_map: Dict[str, str]) -> str:
    return synonym_map.get(normalize(word), normalize(word))

def keyword_set(text: str, synonym_map: Dict[str, str], use_wordnet: bool = True) -> Set[str]:
    base = {canonicalize(w, synonym_map) for w in tokenize(text)}
    if not use_wordnet:
        return base
    return expand_with_wordnet(base)


_wordnet_checked = False
_wordnet_available = False

def expand_with_wordnet(words: Set[str]) -> Set[str]:
    global _wordnet_checked, _wordnet_available
    if not _wordnet_checked:
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets('test')
            _wordnet_available = True
        except Exception:
            print("NLTK or WordNet not found/downloaded. Skipping synonym expansion.")
            _wordnet_available = False
        _wordnet_checked = True
    if not _wordnet_available:
        return words
    from nltk.corpus import wordnet as wn
    expanded = set(words)
    for w in list(words):
        for syn in wn.synsets(w):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if " " not in name and len(name) >= 3:
                    expanded.add(name)
    return expanded

def procedure_signal_count(text: str) -> int:
    if not text:
        return 0
    bullets = len(STEP_BULLET_RE.findall(text))
    seps = len(STEP_SEPARATOR_RE.findall(text))
    verbs = len(set(m.group(1).lower() for m in VERB_RE.finditer(text)))
    seps = min(seps, 8)
    return bullets * 4 + verbs * 2 + seps

def debug_candidate(request_index: int, original_corpus_index: int, query: str, doc_text: str, prompt: str) -> None:
    snippet = doc_text[:200].replace("\n", " ")
    print(
        f"[debug candidate] request={request_index} original={original_corpus_index} "
        f"query='{query[:80]}...' doc='{snippet[:80]}...\n"
        f"--- FULL PROMPT ---\n{prompt}\n--- END FULL PROMPT ---"
    )

def domain_gate_allows(query: str, doc: str, context: PolicyContext) -> bool:
    if not context.domain_keywords:
        return True
    
    q_tokens = keyword_set(query, context.synonym_map, use_wordnet=False)
    if not (q_tokens & context.domain_keywords):
        return True # Query is not specific to the domain, so allow all docs.

    d_tokens = keyword_set(doc, context.synonym_map, use_wordnet=False)
    return bool(d_tokens & context.domain_keywords)

def query_has_intent(query: str, context: PolicyContext) -> bool:
    tokens = keyword_set(query, context.synonym_map, use_wordnet=False)
    return bool(tokens & context.intent_keywords)

def intent_requires_procedure_cap(
    query: str,
    doc: str,
    score: int,
    context: PolicyContext,
) -> Tuple[int, Optional[str]]:
    if not query_has_intent(query, context):
        return score, None
    proc = procedure_signal_count(doc)
    if score == 3 and proc < 2:
        return 2, "three_requires_steps"
    if score == 2 and proc < 1:
        return 1, "two_requires_some_steps"
    return score, None


def log_candidate_inspection(
    request_index: int,
    original_corpus_index: int,
    query: str,
    doc_text: str,
    bucket: int,
    final_score: int,
    tie_breaker: int,
    cap_reason: str,
) -> None:
    q_lower = query.lower()
    doc_lower = doc_text.lower()
    if not any(keyword in q_lower or keyword in doc_lower for keyword in ("faucet", "maintenance")):
        return
    snippet = doc_text.replace("\n", " ")[:120]
    tier = tie_breaker // 100
    print(
        f"[candidate detail] idx={original_corpus_index} query='{query[:40]}...' "
        f"bucket={bucket} final={final_score} tier={tier} reason='{cap_reason or 'none'}' "
        f"doc='{snippet[:80]}...'"
    )


def _keyword_overlap_tier(query: str, doc: str, context: PolicyContext) -> int:
    qk = keyword_set(query, context.synonym_map, use_wordnet=True)
    if not qk:
        return 2
    dk = keyword_set(doc, context.synonym_map, use_wordnet=False)
    overlap = qk & dk
    if not overlap:
        return 0
    if overlap & context.tier_a_keywords:
        return 2
    if overlap & context.tier_b_keywords:
        return 1
    return 1


def _trim_document_for_reranking(doc_text: str) -> str:
    if len(doc_text) > 800:
        return f"{doc_text[:600]} [...] {doc_text[-200:]}"
    return doc_text


async def score_documents_concurrently(
    ollama_client: OllamaClient,
    model: str,
    query: str,
    documents: List[Tuple[int, int, str, int]],
    options: Options,
    context: PolicyContext,
    facts_map: Dict[int, Dict[str, Any]],
) -> Tuple[List[Tuple[int, int, int, int]], List[Dict[str, Any]]]:
    scoring_tasks = []
    gate_stats = {"tier_a": 0, "tier_b": 0, "skipped": 0}

    generate_options: Dict[str, Any] = {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "stream": False,
        "num_predict": 32, # Allow more tokens for JSON
        "stop": ["\n", "}"],
        "repeat_penalty": 1.05,
    }
    if options and options.seed is not None:
        generate_options["seed"] = options.seed
    if options and options.num_ctx is not None:
        generate_options["num_ctx"] = options.num_ctx

    for request_index, original_corpus_index, doc_text, original_len in documents:
        procedural_strength = procedure_signal_count(doc_text)
        tier = _keyword_overlap_tier(query, doc_text, context)
        tie_breaker = tier * 100 + procedural_strength

        # The new gating logic is now handled by the prompt and identity facts.
        # We can still have a hard tier-0 gate for performance.
        if request_index >= 5:
            if tier == 0:
                print(f"Skipping LLM call for idx {original_corpus_index} due to hard gate (tier 0).")
                gate_stats["skipped"] += 1
                scoring_tasks.append(
                    asyncio.sleep(0, result=(request_index, original_corpus_index, 0, tie_breaker, None))
                )
                continue

            if not domain_gate_allows(query, doc_text, context):
                print(f"Domain gate rejected idx {original_corpus_index} for query '{query}'.")
                gate_stats["skipped"] += 1
                scoring_tasks.append(
                    asyncio.sleep(0, result=(request_index, original_corpus_index, 0, tie_breaker, None))
                )
                continue

        gate_stats["tier_a" if tier == 2 else "tier_b"] += 1

        trimmed_doc_text = _trim_document_for_reranking(doc_text)
        facts = facts_map.get(original_corpus_index, {})
        scoring_tasks.append(
            _score_single_document_with_retry(
                ollama_client=ollama_client,
                model=model,
                query=query,
                doc_text=trimmed_doc_text,
                original_len=original_len,
                generate_options=generate_options,
                request_index=request_index,
                original_corpus_index=original_corpus_index,
                tie_breaker=tie_breaker,
                context=context,
                facts=facts,
            )
        )

    all_scored_results = await asyncio.gather(*scoring_tasks)
    all_scored_results.sort(key=lambda x: x[0])
    
    results = [(ri, oi, score, tb) for (ri, oi, score, tb, _warn) in all_scored_results]
    warnings = [_warn for *_, _warn in all_scored_results if _warn is not None]

    print(f"[gate stats] tier_a={gate_stats['tier_a']} tier_b={gate_stats['tier_b']} skipped={gate_stats['skipped']}")
    return results, warnings


async def _score_single_document_with_retry(
    ollama_client: OllamaClient,
    model: str,
    query: str,
    doc_text: str,
    original_len: int,
    generate_options: Dict[str, Any],
    request_index: int,
    original_corpus_index: int,
    tie_breaker: int,
    context: PolicyContext,
    facts: Dict[str, Any],
) -> Tuple[int, int, int, int, Optional[Dict[str, Any]]]:
    async with SEM:
        raw: Optional[str] = None
        for attempt in range(2):
            try:
                prompt = (
                    build_rerank_prompt_single_doc(query, doc_text, context, facts)
                    if attempt == 0
                    else build_retry_score_prompt(query, doc_text, raw or "", context, facts)
                )
                if attempt == 0:
                    debug_candidate(request_index, original_corpus_index, query, doc_text, prompt)

                raw = await asyncio.wait_for(
                    ollama_client.generate(model, prompt, generate_options),
                    timeout=45,
                )

                print(f"\n===== RAW MODEL OUTPUT (idx={original_corpus_index}, attempt={attempt+1}) =====")
                print(repr(raw))
                print("===== END RAW OUTPUT =====")

                bucket = parse_and_bucket(raw)
                if bucket is None:
                    raise ValueError("score parse failed")
                
                final_score, cap_reason = intent_requires_procedure_cap(query, doc_text, bucket, context)

                if final_score != bucket or cap_reason:
                    print(
                        f"[score debug] idx={original_corpus_index} bucket={bucket} final={final_score} "
                        f"reason='{cap_reason or 'none'}'")
                log_candidate_inspection(
                    request_index=request_index,
                    original_corpus_index=original_corpus_index,
                    query=query,
                    doc_text=doc_text,
                    bucket=bucket,
                    final_score=final_score,
                    tie_breaker=tie_breaker,
                    cap_reason=cap_reason if cap_reason else "",
                )
                return (request_index, original_corpus_index, final_score, tie_breaker, None)

            except (ValueError, asyncio.TimeoutError, httpx.ReadTimeout, httpx.ConnectError) as e:
                warning = {
                    "idx": original_corpus_index,
                    "error": str(e),
                    "raw": raw if "raw" in locals() else None
                }
                print(f"Warning: Scoring failed for idx {original_corpus_index}. Reason: {e}")
                if attempt == 0:
                    continue
                return (request_index, original_corpus_index, 0, tie_breaker, warning)
            except Exception as e:
                 warning = {
                    "idx": original_corpus_index,
                    "error": f"An unexpected error occurred: {e}",
                    "raw": raw if "raw" in locals() else None
                }
                 return (request_index, original_corpus_index, 0, tie_breaker, warning)

        warning = {
            "idx": original_corpus_index,
            "error": "Scoring failed after all retries.",
            "raw": raw if "raw" in locals() else None
        }
        return (request_index, original_corpus_index, 0, tie_breaker, warning)