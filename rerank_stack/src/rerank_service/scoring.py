from __future__ import annotations
import asyncio
import httpx
import hashlib
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set
import time

from .ollama_client import OllamaClient
from .policy import PolicyContext
from .schemas import Options
from .parser import parse_and_bucket, parse_fine_score, fallback_fine_score
from .prompt_builder import (
    build_rerank_prompt_single_doc,
    build_retry_score_prompt,
    build_fine_score_prompt,
)

_max_concurrency = int(os.getenv("RERANK_TIE_MAX_CONCURRENCY", "4"))
SEM = asyncio.Semaphore(max(1, _max_concurrency))

# Simple in-memory cache for tie-break outputs: key -> (expires_at, result_dict)
_TIE_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_TIE_CACHE_TTL = float(os.getenv("RERANK_TIE_CACHE_TTL_SEC", "300"))
_CACHE_LOCK = asyncio.Lock()
_TIE_FAILS: List[float] = []
_CB_THRESHOLD = int(os.getenv("RERANK_TIE_CB_THRESHOLD", "5"))
_CB_WINDOW_SEC = float(os.getenv("RERANK_TIE_CB_WINDOW_SEC", "60"))


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
    documents: List[Tuple[int, int, str, int, Optional[float]]],
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

    for request_index, original_corpus_index, doc_text, original_len, _sim in documents:
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


async def fine_score_documents(
    ollama_client: OllamaClient,
    model: str,
    query: str,
    docs: List[Tuple[int, str]],
) -> Dict[int, Dict[str, Any]]:
    """
    Fine-grained scoring for already-accepted docs.
    Returns mapping original_corpus_index -> {fine_score, fine_score_norm, error, latency_ms, raw}.
    """
    results: Dict[int, Dict[str, Any]] = {}
    generate_options: Dict[str, Any] = {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "stream": False,  # rely on non-stream for easier parsing
        "num_predict": 16,  # just enough for "<int>%"
        "repeat_penalty": 1.05,
        # no JSON format; prompt expects plain "<int>%"
    }

    cache_hits = 0
    timeout_budget = float(os.getenv("RERANK_TIE_TIMEOUT_SEC", "30"))
    total_budget = float(os.getenv("RERANK_TIE_TOTAL_TIMEOUT_SEC", "60"))
    doc_clip_chars = int(os.getenv("RERANK_TIE_DOC_CHARS", "1600"))
    metrics = {
        "cache_hits": 0,
        "requested": len(docs),
        "cache_size": 0,
        "ttl_sec": _TIE_CACHE_TTL,
        "parse_errors": 0,
        "retries": 0,
        "retry_errors": 0,
        "timeouts": 0,
        "fallbacks": 0,
        "skipped_budget": 0,
        "circuit_open": 0,
    }

    # Track total budget start
    total_started = time.perf_counter()

    # Circuit breaker: if too many recent failures, skip tie-break for this request.
    async with _CACHE_LOCK:
        now = time.time()
        recent_fails = [t for t in _TIE_FAILS if (now - t) <= _CB_WINDOW_SEC]
        _TIE_FAILS[:] = recent_fails
        cb_open = len(recent_fails) >= _CB_THRESHOLD

    if cb_open:
        metrics["circuit_open"] = 1
        for orig_idx, _doc_text in docs:
            results[orig_idx] = {
                "fine_score": None,
                "fine_score_norm": None,
                "error": "circuit_open",
                "latency_ms": 0,
                "raw": None,
            }
        if os.getenv("RERANK_TIE_METRICS", "").lower() in {"1", "true", "yes"}:
            print(
                "[fine_score metrics]",
                {
                    **metrics,
                    "cache_size": len(_TIE_CACHE),
                },
            )
        return results

    for orig_idx, doc_text in docs:
        if (time.perf_counter() - total_started) > total_budget:
            metrics["skipped_budget"] += 1
            results[orig_idx] = {
                "fine_score": None,
                "fine_score_norm": None,
                "error": "skipped_total_budget",
                "latency_ms": 0,
                "raw": None,
            }
            continue

        doc_text = (doc_text or "")[:doc_clip_chars]
        prompt = build_fine_score_prompt(query, doc_text)
        prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        doc_hash = hashlib.sha1((doc_text or "").encode("utf-8")).hexdigest()
        query_hash = hashlib.sha1((query or "").encode("utf-8")).hexdigest()
        cache_key = f"{model}:{prompt_hash}:{doc_hash}:{query_hash}"

        # Cache lookup
        async with _CACHE_LOCK:
            cached = _TIE_CACHE.get(cache_key)
            if cached and cached[0] > time.time():
                cache_hits += 1
                metrics["cache_hits"] += 1
                results[orig_idx] = cached[1]
                continue
            elif cached:
                _TIE_CACHE.pop(cache_key, None)

        started = time.perf_counter()
        raw: Optional[str] = None
        raw_retry: Optional[str] = None
        error: Optional[str] = None
        fine_score: Optional[int] = None
        try:
            async with SEM:
                raw = await asyncio.wait_for(
                    ollama_client.generate(model, prompt, generate_options),
                    timeout=timeout_budget,
                )
            if not str(raw or "").strip():
                error = "empty_output"
                print(
                    "[fine_score empty_output]",
                    {
                        "orig_idx": orig_idx,
                        "query": query[:80],
                        "doc_head": doc_text[:120],
                        "prompt_sha1": prompt_hash,
                        "prompt_len": len(prompt),
                        "prompt_head": prompt[:300],
                        "options": generate_options,
                    },
                )
            else:
                fine_score, err = parse_fine_score(raw)
                error = f"percent_parse:{err}" if fine_score is None and err else None
                if error and ("percent_parse" in error):
                    metrics["parse_errors"] += 1
                    print(
                        "[fine_score parse_error]",
                        {
                            "orig_idx": orig_idx,
                            "prompt_sha1": prompt_hash,
                            "prompt_head": prompt[:200],
                            "raw_len": len(str(raw)),
                            "raw_head": repr(str(raw)[:200]),
                            "error": error,
                            "options": generate_options,
                            "parser": "percent_line",
                        },
                    )
                    # Retry once with a minimal prompt to coerce a numeric answer
                    retry_prompt = (
                        "Return ONLY one integer 0-100 followed by a percent sign, like 70%.\n"
                        "No explanations. One line. Output now:"
                    )
                    retry_options = {
                        **generate_options,
                        "num_predict": 12,
                        "stop": ["\n"],
                    }
                    try:
                        metrics["retries"] += 1
                        async with SEM:
                            raw_retry = await asyncio.wait_for(
                                ollama_client.generate(model, retry_prompt, retry_options),
                                timeout=timeout_budget / 2,
                            )
                        fine_score_retry, err_retry = parse_fine_score(raw_retry)
                        if fine_score_retry is not None:
                            fine_score = fine_score_retry
                            error = None
                        else:
                            error = f"percent_retry_parse:{err_retry}"
                            metrics["retry_errors"] += 1
                            print(
                                "[fine_score retry_parse_error]",
                                {
                                    "orig_idx": orig_idx,
                                    "raw_retry_head": repr(str(raw_retry)[:200]),
                                    "error": error,
                                    "options": retry_options,
                                    "parser": "percent_line_retry",
                                },
                            )
                    except asyncio.TimeoutError:
                        error = "percent_retry_timeout"
                        metrics["timeouts"] += 1
                    except Exception as exc_retry:
                        error = f"percent_retry_exc:{exc_retry}"
        except asyncio.TimeoutError:
            error = "fine_score_timeout"
            metrics["timeouts"] += 1
        except Exception as exc:
            error = f"fine_score_exc:{exc}"
        latency_ms = int((time.perf_counter() - started) * 1000)
        result_payload = {
            "fine_score": fine_score,
            "fine_score_norm": (fine_score / 100.0) if fine_score is not None else None,
            "error": error,
            "latency_ms": latency_ms,
            "raw": raw_retry or raw,
        }
        if error and ("fallback" in (error or "")):
            metrics["fallbacks"] += 1
        results[orig_idx] = result_payload
        # cache set
        async with _CACHE_LOCK:
            _TIE_CACHE[cache_key] = (time.time() + _TIE_CACHE_TTL, result_payload)
            if error:
                _TIE_FAILS.append(time.time())

    if os.getenv("RERANK_TIE_METRICS", "").lower() in {"1", "true", "yes"}:
        print(
            "[fine_score metrics]",
            {
                **metrics,
                "cache_size": len(_TIE_CACHE),
            },
        )
    return results
