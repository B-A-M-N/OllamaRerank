import argparse
import asyncio
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import numpy as np
from scipy.spatial.distance import cosine

# Add the 'src' directory to sys.path for module imports
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent / "src"))

from rerank_demo.build_index import OllamaEmbeddingClient  # Reusing the client from build_index
from rerank_service.schemas import Document, RerankRequest

# Assuming the Rerank API is running at this address
RERANK_API_URL = os.environ.get("RERANK_API_URL", "http://127.0.0.1:8000/api/rerank")
EMBEDDING_MODEL = "nomic-embed-text"
TOP_K_RETRIEVAL = int(os.environ.get("TOP_K_RETRIEVAL", 10))  # Number of documents to retrieve before reranking

# --- Evaluation Set ---
# Each entry maps a query to all acceptable documents for that intent.
EVAL_SET = [
    {"query": "replace shower cartridge", "relevant_any": {0, 2}},
    {"query": "car maintenance", "relevant_any": {1, 5, 9}},
    {"query": "bicycle repair", "relevant_any": {3, 7}},
    {"query": "leaky faucet fix", "relevant_any": {4}},
    {"query": "maintenance", "relevant_any": {3, 5}},
    {"query": "fix leaking faucet", "relevant_any": {4}},
]

ZERO_TIER = {
    "item_mismatch": 0,
    "no_item_found": 1,
    "domain_mismatch": 2,
    None: 9,
}


def _zero_tier_from_facts(facts: Dict[str, Any]) -> int:
    return ZERO_TIER.get(facts.get("zero_reason"), 9)


def _decision_from_doc(doc: Dict[str, Any], facts: Dict[str, Any]) -> str:
    decision = facts.get("decision") or doc.get("decision")
    if decision:
        return decision
    if facts.get("rerank_mode") == "fallback_intent_similarity":
        return "ACCEPT"
    if doc.get("score", 0) > 0:
        return "ACCEPT"
    return f"REJECT_TIER_{_zero_tier_from_facts(facts)}"


def _score_norm(doc: Dict[str, Any], facts: Dict[str, Any], max_score: Optional[float]) -> Optional[float]:
    for key in ("score_norm", "rank_score_norm", "fine_score_norm"):
        val = doc.get(key)
        if val is None:
            val = facts.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    if max_score and max_score > 0 and isinstance(doc.get("score"), (int, float)):
        return doc["score"] / max_score
    return None


def _format_match(label: str, value: Optional[str], matched: Optional[bool]) -> str:
    if matched is True:
        suffix = "+"
    elif matched is False:
        suffix = "x"
    else:
        suffix = "?"
    return f"{label}={value or '-'}{suffix}"


def _format_tie_info(doc: Dict[str, Any], facts: Dict[str, Any]) -> Optional[str]:
    tie_used = doc.get("tie_breaker_used") or facts.get("tie_breaker_used")
    tie_error = doc.get("tie_breaker_error") or facts.get("tie_breaker_error")
    tie_latency = doc.get("tie_breaker_latency_ms") or facts.get("tie_breaker_latency_ms")
    fine_score = doc.get("fine_score") or facts.get("fine_score")
    fine_norm = doc.get("fine_score_norm") or facts.get("fine_score_norm")

    if not tie_used and not tie_error and tie_latency is None:
        return None

    if tie_error:
        label = f"ERR({tie_error})"
    elif fine_score is not None:
        label = f"{fine_score}%"
    elif fine_norm is not None:
        label = f"{round(fine_norm * 100)}%"
    elif tie_used:
        label = "used"
    else:
        label = "pending"

    if tie_latency is not None:
        return f"{label} ({int(tie_latency)}ms)"
    return label


def _summarize_ties(docs: List[Dict[str, Any]]) -> Tuple[int, int, Optional[int]]:
    used = 0
    parse_failed = 0
    latencies: List[int] = []
    for doc in docs:
        facts = doc.get("facts") or {}
        tie_used = doc.get("tie_breaker_used") or facts.get("tie_breaker_used")
        tie_error = doc.get("tie_breaker_error") or facts.get("tie_breaker_error")
        tie_latency = doc.get("tie_breaker_latency_ms") or facts.get("tie_breaker_latency_ms")
        if tie_used:
            used += 1
        if tie_error and "parse" in str(tie_error).lower():
            parse_failed += 1
        if tie_latency is not None:
            try:
                latencies.append(int(tie_latency))
            except (TypeError, ValueError):
                pass
    avg_latency = None
    if latencies:
        avg_latency = int(sum(latencies) / len(latencies))
    return used, parse_failed, avg_latency


def _print_compact_results(
    query: str,
    policy_id: Optional[str],
    rerank_mode: str,
    model_used: str,
    final_docs: List[Dict[str, Any]],
    show_facts: bool,
    explain_rank: Optional[int],
    endpoint: str,
    ambiguous_reason: str,
) -> None:
    max_score = max([d.get("score", 0.0) for d in final_docs], default=0.0)

    def _snippet(text: str, limit: int = 80) -> str:
        t = (text or "").replace("\n", " ").strip()
        return (t[: limit - 3] + "...") if len(t) > limit else t

    print("\n--- Rerank Results ---")
    mode = "AMBIGUOUS" if rerank_mode.startswith("fallback") else "SPECIFIC"
    mode_detail = f"{mode}{f' ({ambiguous_reason})' if ambiguous_reason else ''}"
    print(f"Query: {query}")
    print(f"Mode: {mode_detail} | Policy: {policy_id or '-'}")
    print(f"Endpoint: {endpoint} | Model: {model_used}")

    accepted_lines: List[str] = []
    kept_lines: List[str] = []
    filtered: Dict[str, List[str]] = {}

    for rank, doc in enumerate(final_docs, start=1):
        facts = doc.get("facts") or {}
        decision = _decision_from_doc(doc, facts)
        score = doc.get("score")
        score_block = "score=?"
        if isinstance(score, (int, float)):
            score_block = f"score={score:.2f}"
            norm = _score_norm(doc, facts, max_score)
            if norm is not None:
                score_block += f" (norm={norm:.2f})"
        item_flag = _format_match("item", facts.get("query_item"), facts.get("same_item"))
        domain_flag = _format_match("domain", facts.get("doc_domain"), facts.get("same_domain"))
        kind = facts.get("doc_kind") or facts.get("mode")
        intent_fit = facts.get("intent_fit")
        sim = doc.get("similarity")
        tie_text = _format_tie_info(doc, facts)

        parts = [
            f"#{rank}",
            f"idx={doc.get('original_corpus_index')}",
            score_block,
            decision,
            item_flag,
            domain_flag,
        ]
        if kind:
            parts.append(f"kind={kind}")
        if intent_fit is not None:
            try:
                parts.append(f"intent_fit={float(intent_fit):.2f}")
            except (TypeError, ValueError):
                pass
        if isinstance(sim, (int, float)):
            parts.append(f"sim={sim:.3f}")
        if tie_text:
            parts.append(f"tie={tie_text}")

        snippet = _snippet(doc.get("document") or "")
        line = "  ".join(parts + [f'text="{snippet}"'])
        is_accept = decision == "ACCEPT"
        is_keep = decision == "KEEP_LOW_SIGNAL"

        if is_accept:
            accepted_lines.append(line)
        elif is_keep:
            kept_lines.append(line)
        else:
            reason = (facts.get("zero_reason") or "unknown").replace("_", " ")
            filtered.setdefault(reason, []).append(
                f"idx={doc.get('original_corpus_index')} (\"{(doc.get('document') or '')[:60].replace(chr(10), ' ')}...\")"
            )

        if show_facts or (explain_rank is not None and explain_rank == rank):
            print(line)
            print("   FACTS:", json.dumps(facts, indent=2, sort_keys=True, default=str))

    print("\nTop Results")
    for ln in accepted_lines:
        print(ln)
    if kept_lines:
        if accepted_lines:
            print("\nKept Low-Signal")
        for ln in kept_lines:
            print(ln)

    if filtered:
        print("\nFiltered Out")
        for reason, entries in filtered.items():
            print(f"- {reason}: " + ", ".join(entries))

    accepted_count = len(accepted_lines)
    kept_count = len(kept_lines)
    total = len(final_docs)
    reject_by_tier = {0: 0, 1: 0, 2: 0, 9: 0}
    for doc in final_docs:
        facts = doc.get("facts") or {}
        decision = _decision_from_doc(doc, facts)
        if decision in {"ACCEPT", "KEEP_LOW_SIGNAL"}:
            continue
        tier = _zero_tier_from_facts(facts)
        reject_by_tier[tier] = reject_by_tier.get(tier, 0) + 1
    other_reject = total - accepted_count - kept_count - sum(reject_by_tier.values())
    used, parse_failed, avg_latency = _summarize_ties(final_docs)

    print("\nDecision Breakdown:")
    print(f"- Accepted: {accepted_count} / {total}")
    print(f"- Rejected: tier0={reject_by_tier.get(0,0)} tier1={reject_by_tier.get(1,0)} tier2={reject_by_tier.get(2,0)} other={max(other_reject, 0)}")
    print(f"- Kept low-signal: {kept_count}")
    tie_line = f"- Tie-breakers: used={used} parse_failed={parse_failed}"
    if avg_latency is not None:
        tie_line += f" avg_latency={avg_latency}ms"
    print(tie_line)

    if explain_rank and (explain_rank < 1 or explain_rank > len(final_docs)):
        print(f"\nExplain requested for rank {explain_rank}, but only {len(final_docs)} results returned.")


def acc_at_1_any(top_doc_index: Optional[int], relevant_any: Set[int]) -> int:
    if top_doc_index is None:
        return 0
    return 1 if top_doc_index in relevant_any else 0


def compute_mrr(ranked_indices: List[int], relevant_any: Set[int]) -> float:
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx in relevant_any:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_indices: List[int], relevant_any: Set[int], k: int = 3) -> int:
    top_k = ranked_indices[:k]
    return int(bool(set(top_k) & relevant_any))


def recall_all_at_k(ranked_indices: List[int], relevant_any: Set[int], k: int = 3) -> float:
    if not relevant_any:
        return 0.0
    top_k = ranked_indices[:k]
    hits = len(set(top_k) & relevant_any)
    return hits / len(relevant_any)


def evaluate_rankings(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    reranked_docs: List[Dict[str, Any]],
    relevant_any: Set[int],
):
    label = "Accuracy@1_any" if len(relevant_any) > 1 else "Accuracy@1"
    initial_ranked_indices = [doc["index"] for doc in retrieved_docs]
    reranked_indices = [doc["original_corpus_index"] for doc in reranked_docs]

    initial_accuracy = acc_at_1_any(initial_ranked_indices[0] if initial_ranked_indices else None, relevant_any)
    reranked_accuracy = acc_at_1_any(reranked_indices[0] if reranked_indices else None, relevant_any)
    initial_mrr = compute_mrr(initial_ranked_indices, relevant_any)
    reranked_mrr = compute_mrr(reranked_indices, relevant_any)
    initial_recall3 = recall_at_k(initial_ranked_indices, relevant_any, k=3)
    reranked_recall3 = recall_at_k(reranked_indices, relevant_any, k=3)
    initial_recall_all = recall_all_at_k(initial_ranked_indices, relevant_any, k=3)
    reranked_recall_all = recall_all_at_k(reranked_indices, relevant_any, k=3)

    print("\nMetrics:")
    print(f"- {label}: initial={initial_accuracy} -> reranked={reranked_accuracy}")
    print(f"- MRR: initial={initial_mrr:.2f} -> reranked={reranked_mrr:.2f}")
    print(f"- Recall@3_any: initial={initial_recall3} -> reranked={reranked_recall3}")
    print(f"- Recall@3_all: initial={initial_recall_all:.2f} -> reranked={reranked_recall_all:.2f}")

    return initial_accuracy, reranked_accuracy


async def search_and_rerank(
    query: str,
    relevant_docs_indices: Optional[Set[int]] = None,
    policy_id: Optional[str] = None,
    show_facts: bool = False,
    explain_rank: Optional[int] = None,
    show_debug: bool = False,
):
    print(f"Searching and reranking for query: '{query}'")

    # Construct paths relative to the script's location
    script_dir = Path(__file__).resolve().parent
    rerank_stack_root = script_dir.parent.parent  # Go up src/ then rerank_demo/
    corpus_file_path = rerank_stack_root / "corpus.json"
    index_file_path = rerank_stack_root / "index.npz"

    # 1. Load corpus and embeddings
    try:
        with open(corpus_file_path, "r") as f:
            corpus = json.load(f)
        embeddings_data = np.load(index_file_path)
        corpus_embeddings = embeddings_data["embeddings"]
    except FileNotFoundError:
        print(f"Error: {corpus_file_path} or {index_file_path} not found. Please run build_index.py first.")
        return 0, 0  # Return 0,0 for accuracies if files not found
    except Exception as e:
        print(f"Error loading corpus or embeddings: {e}")
        return 0, 0  # Return 0,0 for accuracies on error

    # 2. Compute query embedding
    embedding_client = OllamaEmbeddingClient()
    async with embedding_client:
        try:
            query_embedding_list = await embedding_client.embed(EMBEDDING_MODEL, [query])
            query_embedding = np.array(query_embedding_list[0], dtype=np.float32)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return 0, 0  # Return 0,0 for accuracies on error

    # query_embedding is expected to be 1D here. No reshape needed.

    # 3. Cosine similarity for first-pass retrieval
    similarities = 1 - np.array([cosine(query_embedding, doc_embed) for doc_embed in corpus_embeddings])

    # Get top-K candidates
    top_k_indices = np.argsort(similarities)[::-1][:TOP_K_RETRIEVAL]

    retrieved_documents = [{"index": int(i), "score": float(similarities[i]), "document": corpus[i]} for i in top_k_indices]

    print("\n--- Initial Retrieval (Before Rerank) ---")
    for i, doc in enumerate(retrieved_documents):
        print(f"{i+1}. Index: {doc['index']}, Similarity: {doc['score']:.4f}, Doc: {doc['document'][:100]}...")

    # 4. Decide whether to call /api/rerank service or fall back for ambiguous/short queries.
    initial_scores_map = {doc["index"]: doc["score"] for doc in retrieved_documents}

    def detect_ambiguity(q: str) -> Tuple[bool, str]:
        toks = set(re.findall(r"[a-zA-Z]+", q.lower()))
        if not toks:
            return True, "no_tokens"
        if len(toks) <= 1:
            if toks & {"maintenance", "help", "issue", "problem"}:
                return True, "ambiguous_trigger"
            return True, "too_short"
        return False, ""

    is_ambiguous, ambiguous_reason = detect_ambiguity(query)
    # Default to skipping rerank API for ambiguous queries; allow opt-out via env.
    skip_ambiguous = os.getenv("RERANK_SKIP_AMBIGUOUS", "1").lower() in {"1", "true", "yes"}

    final_reranked_docs: List[Dict[str, Any]] = []
    rerank_mode_used = "rerank_api"
    model_used = "B-A-M-N/qwen3-reranker-0.6b-fp16"

    if is_ambiguous and skip_ambiguous:
        # Cheap rerank: intent-aware similarity without hitting the rerank API.
        print(f"\nSkipping rerank API: ambiguous/short query ({ambiguous_reason}). Using intent-aware similarity fallback.\n")

        maint_terms = {"maintenance", "maintain", "maintaining", "service", "servicing", "care", "upkeep", "routine", "schedule", "interval", "checklist"}
        query_tokens = set(re.findall(r"[a-zA-Z]+", query.lower()))
        query_intent = "OVERVIEW" if query_tokens & maint_terms else "UNKNOWN"

        # Tiny domain detector for diversification
        def detect_domain(text: str) -> str:
            t = text.lower()
            if any(w in t for w in ["shower", "faucet", "plumbing", "pipe", "cartridge", "valve", "water heater"]):
                return "plumbing"
            if any(w in t for w in ["car", "engine", "oil", "brake", "spark", "civic", "automotive"]):
                return "automotive"
            if any(w in t for w in ["bicycle", "bike", "chain", "derailleur", "cassette", "tire"]):
                return "bicycle"
            return "unknown"

        def simple_doc_kind(text: str) -> str:
            t = text.lower()
            toks = set(re.findall(r"[a-zA-Z]+", t))
            verb_hits = sum(1 for w in ["turn", "remove", "install", "replace", "tighten", "loosen", "use", "check", "inspect", "clean", "lube", "lubricate"] if w in toks)
            if t.startswith("how to") or "steps to" in t or verb_hits >= 2:
                return "PROCEDURAL"
            if any(w in toks for w in maint_terms):
                return "OVERVIEW"
            if any(phrase in t for phrase in ["types of", "understanding", "overview", "guide to"]):
                return "INFORMATIONAL"
            if "symptoms" in t or "signs of" in t:
                return "DIAGNOSTIC"
            return "UNKNOWN"

        def intent_fit(q_intent: str, d_kind: str) -> float:
            matrix = {
                "OVERVIEW": {"OVERVIEW": 1.0, "PROCEDURAL": 0.8, "INFORMATIONAL": 0.6, "DIAGNOSTIC": 0.6, "UNKNOWN": 0.7},
                "UNKNOWN": {"PROCEDURAL": 0.8, "OVERVIEW": 0.8, "INFORMATIONAL": 0.7, "DIAGNOSTIC": 0.7, "UNKNOWN": 0.7},
            }
            return matrix.get(q_intent, matrix["UNKNOWN"]).get(d_kind, 0.7)

        def lexical_bonus(doc_text: str) -> float:
            toks = set(re.findall(r"[a-zA-Z]+", doc_text.lower()))
            bonus = 0.0
            if toks & maint_terms:
                bonus += 0.2
            if "how" in toks or "steps" in toks:
                bonus += 0.1
            return min(0.3, bonus)

        scored_fallback = []
        for doc in retrieved_documents:
            doc_tokens = set(re.findall(r"[a-zA-Z]+", doc["document"].lower()))
            overlap = len(doc_tokens & query_tokens)
            d_kind = simple_doc_kind(doc["document"])
            fit = intent_fit(query_intent, d_kind)
            lex = lexical_bonus(doc["document"])
            # Cheap final score: intent-fit weighted similarity + small lexical/overlap nudges.
            score = (0.7 * doc["score"] * fit) + (0.15 * overlap) + (0.15 * lex)
            domain = detect_domain(doc["document"])
            scored_fallback.append({
                "original_corpus_index": doc["index"],
                "score": score,
                "similarity": doc["score"],
                "document": doc["document"],
                "domain": domain,
                "facts": {
                    "rerank_mode": "fallback_intent_similarity",
                    "skip_reason": ambiguous_reason,
                    "doc_kind": d_kind,
                    "intent_fit": fit,
                    "overlap": overlap,
                    "lexical_bonus": lex,
                    "domain": domain,
                    "decision": "ACCEPT",
                }
            })

        # Diversify: take top-1 from each domain present in top candidates (up to 3 domains), then fill by score.
        scored_fallback.sort(key=lambda r: r["score"], reverse=True)
        domains_order = []
        for r in scored_fallback:
            if r["domain"] not in domains_order:
                domains_order.append(r["domain"])
        picked: List[Dict[str, Any]] = []
        seen_domains = set()
        for dom in domains_order:
            top_dom = [r for r in scored_fallback if r["domain"] == dom]
            if top_dom:
                picked.append(top_dom[0])
                seen_domains.add(dom)
                if len(picked) >= 3:  # grab at most 3 diversified heads
                    break
        # Fill remaining slots by score, skipping already picked docs
        picked_ids = {r["original_corpus_index"] for r in picked}
        for r in scored_fallback:
            if r["original_corpus_index"] in picked_ids:
                continue
            picked.append(r)

        if show_debug:
            print("AMBIGUOUS DEBUG (top 5):", [
                (
                    r["original_corpus_index"],
                    f"{r['similarity']:.3f}",
                    r["domain"],
                    (r["facts"] or {}).get("doc_kind"),
                    (r["facts"] or {}).get("intent_fit"),
                    f"{r['score']:.3f}",
                )
                for r in picked[:5]
            ])

        final_reranked_docs = picked
        rerank_mode_used = "fallback_intent_similarity"
        model_used = "fallback_intent_similarity"
    else:
        rerank_request_docs = [
            Document(text=doc["document"], original_index=doc["index"], similarity=doc["score"])
            for doc in retrieved_documents
        ]
        rerank_payload = RerankRequest(
            model="B-A-M-N/qwen3-reranker-0.6b-fp16",
            query=query,
            documents=rerank_request_docs,
            return_documents=False,
            policy_id=policy_id
        )

        rerank_timeout = httpx.Timeout(connect=5.0, read=600.0, write=300.0, pool=5.0)
        rerank_client = httpx.AsyncClient(timeout=rerank_timeout)
        try:
            print(f"\nCalling Rerank API at: {RERANK_API_URL}")
            rerank_response = await rerank_client.post(RERANK_API_URL, json=rerank_payload.model_dump())
            rerank_response.raise_for_status()
            rerank_data = rerank_response.json()
            reranked_results: List[Dict[str, Any]] = rerank_data.get("results", [])

            for r in reranked_results:
                r["initial_score"] = initial_scores_map.get(r.get("original_corpus_index"), 0.0)

            def _to_float(x):
                if x is None:
                    return float("-inf")
                if isinstance(x, (int, float)):
                    return float(x)
                s = str(x).strip()
                if s.isdigit():
                    return float(s)
                m = re.search(r"\b(3|2|1|0)\b", s)
                if m:
                    return float(m.group(1))
                return float("-inf")

            for r in reranked_results:
                r["score"] = _to_float(r.get("score", 0.0))

            if show_debug:
                print("DEBUG before sort:", [
                    (
                        r.get("original_corpus_index"),
                        r.get("score"),
                        (r.get("facts") or {}).get("rank_score"),
                        r.get("initial_score"),
                    )
                    for r in reranked_results
                ])
            # Partition: keep >0 ahead of 0s; sort positives by score (already refined in API) then sim.
            positives = [r for r in reranked_results if (r.get("facts") or {}).get("decision") == "ACCEPT"]
            keeps = [r for r in reranked_results if (r.get("facts") or {}).get("decision") == "KEEP_LOW_SIGNAL"]
            zeros = [r for r in reranked_results if (r.get("facts") or {}).get("decision") not in {"ACCEPT", "KEEP_LOW_SIGNAL"}]

            def _zero_tier(r):
                facts = r.get("facts") or {}
                return _zero_tier_from_facts(facts)

            zeros.sort(key=lambda r: (_zero_tier(r), -r.get("initial_score", 0.0), r.get("original_corpus_index", 0)))

            if show_debug:
                print("DEBUG after sort: ", [
                    (
                        r.get("original_corpus_index"),
                        r.get("score"),
                        (r.get("facts") or {}).get("rank_score"),
                        r.get("initial_score"),
                        (r.get("facts") or {}).get("decision") or ("ACCEPT" if r.get("score", 0.0) > 0 else f"REJECT_TIER_{_zero_tier(r)}"),
                        (r.get("facts") or {}).get("zero_reason"),
                    )
                    for r in positives + keeps + zeros
                ])
            reranked_results = positives + keeps + zeros

            scores = [r["score"] for r in reranked_results]
            if scores:
                uniq = {round(s, 8) for s in scores}
                if len(uniq) <= 1:
                    print("\n!!! RERANKER FLATLINE: all scores identical. Rerank is currently a no-op.\n")

            for result in reranked_results:
                result_facts = result.get("facts") or {}
                result_facts["rerank_mode"] = "rerank_api"
                final_reranked_docs.append({
                    "original_corpus_index": result["original_corpus_index"],
                    "score": result["score"],
                    "score_norm": result.get("score_norm"),
                    "fine_score": result.get("fine_score"),
                    "fine_score_norm": result.get("fine_score_norm"),
                    "tie_breaker_used": result.get("tie_breaker_used"),
                    "tie_breaker_model": result.get("tie_breaker_model"),
                    "tie_breaker_prompt_id": result.get("tie_breaker_prompt_id"),
                    "tie_breaker_error": result.get("tie_breaker_error"),
                    "tie_breaker_latency_ms": result.get("tie_breaker_latency_ms"),
                    "decision": result.get("decision"),
                    "similarity": result.get("initial_score"),
                    "document": corpus[result["original_corpus_index"]],
                    "facts": result_facts,
                })

        except httpx.HTTPStatusError as e:
            print("Network error calling Rerank API:")
            print(repr(e))
            try:
                print("Response body:")
                print(e.response.text)
            except Exception:
                pass
            traceback.print_exc()
            raise RuntimeError(f"Rerank API error: {e}") from e
        except httpx.RequestError as e:
            print("Network error calling Rerank API:")
            print(repr(e))
            traceback.print_exc()
            raise RuntimeError(f"Rerank API request failed: {e}") from e
        except Exception as e:
            print(f"An unexpected error occurred during reranking: {e}")
            traceback.print_exc()
            raise
        finally:
            await rerank_client.aclose()

    for doc in final_reranked_docs:
        if "similarity" not in doc:
            doc["similarity"] = initial_scores_map.get(doc.get("original_corpus_index"))
        facts = doc.get("facts") or {}
        facts.setdefault("decision", _decision_from_doc(doc, facts))

    _print_compact_results(
        query=query,
        policy_id=policy_id,
        rerank_mode=rerank_mode_used if final_reranked_docs else "unknown",
        model_used=model_used,
        final_docs=final_reranked_docs,
        show_facts=show_facts,
        explain_rank=explain_rank,
        endpoint=RERANK_API_URL,
        ambiguous_reason=ambiguous_reason,
    )

    if relevant_docs_indices:
        return evaluate_rankings(query, retrieved_documents, final_reranked_docs, relevant_docs_indices)

    return 0, 0  # Default return for non-eval mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Search + rerank demo runner.")
    parser.add_argument("query", nargs="*", help="Query text to run (omit for none)")
    parser.add_argument("--eval", action="store_true", help="Run the built-in evaluation set")
    parser.add_argument("--policy", help="Policy id to send with the rerank request")
    parser.add_argument("--show-facts", action="store_true", help="Print full FACTS for every result")
    parser.add_argument("--explain", type=int, help="Print FACTS for the given rank (1-based)")
    parser.add_argument("--debug", action="store_true", help="Show debug sorting output")
    args = parser.parse_args()

    if args.eval:
        total_initial_accuracy = 0
        total_reranked_accuracy = 0
        for eval_case in EVAL_SET:
            initial_acc, reranked_acc = asyncio.run(
                search_and_rerank(
                    eval_case["query"],
                    relevant_docs_indices=eval_case["relevant_any"],
                    policy_id=args.policy,
                    show_facts=args.show_facts,
                    explain_rank=args.explain,
                    show_debug=args.debug,
                )
            )
            total_initial_accuracy += initial_acc
            total_reranked_accuracy += reranked_acc

        print("\n=== Overall Evaluation Results ===")
        avg_initial = total_initial_accuracy / len(EVAL_SET)
        avg_reranked = total_reranked_accuracy / len(EVAL_SET)
        print(f"Average Initial Retrieval Accuracy@1_any: {avg_initial:.2f}")
        print(f"Average Reranked Accuracy@1_any: {avg_reranked:.2f}")
    elif args.query:
        query_text = " ".join(args.query)
        asyncio.run(
            search_and_rerank(
                query_text,
                policy_id=args.policy,
                show_facts=args.show_facts,
                explain_rank=args.explain,
                show_debug=args.debug,
            )
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
