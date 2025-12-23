import sys
from pathlib import Path

# Add the 'src' directory to sys.path for module imports
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent / "src"))

import httpx
import json
import numpy as np
import asyncio
import re
from typing import List, Dict, Any, Optional, Set, Tuple
import os # Added os import
import traceback # Added traceback import

from rerank_service.schemas import RerankRequest, Document
from rerank_demo.build_index import OllamaEmbeddingClient # Reusing the client from build_index
from scipy.spatial.distance import cosine # Using scipy for cosine distance

# Assuming the Rerank API is running at this address
RERANK_API_URL = os.environ.get("RERANK_API_URL", "http://127.0.0.1:8000/api/rerank")
EMBEDDING_MODEL = "nomic-embed-text"
TOP_K_RETRIEVAL = int(os.environ.get("TOP_K_RETRIEVAL", 10)) # Number of documents to retrieve before reranking

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

    print(f"\n--- Evaluation for Query: '{query}' ---")
    initial_accuracy = acc_at_1_any(initial_ranked_indices[0] if initial_ranked_indices else None, relevant_any)
    reranked_accuracy = acc_at_1_any(reranked_indices[0] if reranked_indices else None, relevant_any)
    print(f"Initial Retrieval {label}: {initial_accuracy}")
    print(f"Reranked {label}: {reranked_accuracy}")
    print(f"Initial MRR: {compute_mrr(initial_ranked_indices, relevant_any):.2f}")
    print(f"Reranked MRR: {compute_mrr(reranked_indices, relevant_any):.2f}")
    print(f"Initial Recall@3_any: {recall_at_k(initial_ranked_indices, relevant_any, k=3)}")
    print(f"Reranked Recall@3_any: {recall_at_k(reranked_indices, relevant_any, k=3)}")
    print(f"Initial Recall@3_all: {recall_all_at_k(initial_ranked_indices, relevant_any, k=3):.2f}")
    print(f"Reranked Recall@3_all: {recall_all_at_k(reranked_indices, relevant_any, k=3):.2f}")
    print("-" * 40)

    return initial_accuracy, reranked_accuracy


async def search_and_rerank(query: str, relevant_docs_indices: Optional[Set[int]] = None, policy_id: Optional[str] = None):
    print(f"Searching and reranking for query: '{query}'")

    # Construct paths relative to the script's location
    script_dir = Path(__file__).resolve().parent
    rerank_stack_root = script_dir.parent.parent # Go up src/ then rerank_demo/
    corpus_file_path = rerank_stack_root / "corpus.json"
    index_file_path = rerank_stack_root / "index.npz"
    docs_file_path = rerank_stack_root / "docs.json" # build_index.py saves to docs.json

    # 1. Load corpus and embeddings
    try:
        with open(corpus_file_path, "r") as f:
            corpus = json.load(f)
        embeddings_data = np.load(index_file_path)
        corpus_embeddings = embeddings_data["embeddings"]
    except FileNotFoundError:
        print(f"Error: {corpus_file_path} or {index_file_path} not found. Please run build_index.py first.")
        return 0, 0 # Return 0,0 for accuracies if files not found
    except Exception as e:
        print(f"Error loading corpus or embeddings: {e}")
        return 0, 0 # Return 0,0 for accuracies on error

    # 2. Compute query embedding
    embedding_client = OllamaEmbeddingClient()
    async with embedding_client:
        try:
            query_embedding_list = await embedding_client.embed(EMBEDDING_MODEL, [query])
            query_embedding = np.array(query_embedding_list[0], dtype=np.float32)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return 0, 0 # Return 0,0 for accuracies on error
    
    # query_embedding is expected to be 1D here. No reshape needed.

    # 3. Cosine similarity for first-pass retrieval
    # Assuming embeddings are L2-normalized (common for embedding models)
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
            # Single-word queries like "maintenance" are too broad.
            if toks & {"maintenance", "help", "issue", "problem"}:
                return True, "ambiguous_trigger"
            return True, "too_short"
        # Two or more lexical tokens -> treat as anchored enough to rerank.
        return False, ""

    is_ambiguous, ambiguous_reason = detect_ambiguity(query)

    final_reranked_docs = []

    if is_ambiguous:
        print(f"\nSkipping rerank API: ambiguous/short query ({ambiguous_reason}). Using similarity fallback.\n")
        for doc in retrieved_documents:
            doc_tokens = set(re.findall(r"[a-zA-Z]+", doc["document"].lower()))
            overlap = len(doc_tokens & set(re.findall(r"[a-zA-Z]+", query.lower())))
            score = (0.85 * doc["score"]) + (0.15 * overlap)
            final_reranked_docs.append({
                "original_corpus_index": doc["index"],
                "score": score,
                "document": doc["document"],
                "facts": {
                    "rerank_mode": "fallback_similarity",
                    "skip_reason": ambiguous_reason,
                    "overlap": overlap,
                }
            })
        final_reranked_docs.sort(key=lambda r: r["score"], reverse=True)
    else:
        rerank_request_docs = [
            Document(text=doc["document"], original_index=doc["index"])
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

            print("DEBUG before sort:", [(r.get("original_corpus_index"), r.get("score"), r.get("initial_score")) for r in reranked_results])
            # Partition: keep >0 ahead of 0s; sort positives by score then sim.
            positives = [r for r in reranked_results if r.get("score", 0.0) > 0]
            zeros = [r for r in reranked_results if r.get("score", 0.0) <= 0]
            positives.sort(key=lambda r: (r.get("score", 0.0), r.get("initial_score", 0.0)), reverse=True)

            def _zero_tier(r):
                facts = r.get("facts") or {}
                same_domain = facts.get("same_domain", False)
                zr = facts.get("zero_reason")
                if same_domain and zr in {"no_item_found", "item_mismatch"}:
                    return 0  # close but wrong/absent item
                if same_domain:
                    return 1  # in-domain generic
                return 2  # clear domain mismatch

            # Within zeros: tier by reason, then by similarity.
            zeros.sort(key=lambda r: (_zero_tier(r), -r.get("initial_score", 0.0)))
            reranked_results = positives + zeros
            print("DEBUG after sort: ", [(r.get("original_corpus_index"), r.get("score"), r.get("initial_score")) for r in reranked_results])

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
                    "document": corpus[result["original_corpus_index"]],
                    "facts": result_facts
                })

            print("\n--- Reranked Results (After Rerank) ---")
            divider_printed = False
            for i, doc in enumerate(final_reranked_docs):
                if not divider_printed and doc.get("score", 0.0) <= 0:
                    print("---- Below: score==0 (rejected/low-signal) ----")
                    divider_printed = True
                print(f"{i+1}. Original Index: {doc['original_corpus_index']}, Rerank Score: {doc['score']:.4f}, Doc: {doc['document'][:100]}...")
                if doc.get("facts"):
                    print(f"   FACTS: {doc['facts']}")

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
    
    if relevant_docs_indices:
        return evaluate_rankings(query, retrieved_documents, final_reranked_docs, relevant_docs_indices)
    
    return 0, 0 # Default return for non-eval mode


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        total_initial_accuracy = 0
        total_reranked_accuracy = 0
        for eval_case in EVAL_SET:
            initial_acc, reranked_acc = asyncio.run(
                search_and_rerank(
                    eval_case["query"],
                    relevant_docs_indices=eval_case["relevant_any"],
                )
            )
            total_initial_accuracy += initial_acc
            total_reranked_accuracy += reranked_acc

        print("\n=== Overall Evaluation Results ===")
        avg_initial = total_initial_accuracy / len(EVAL_SET)
        avg_reranked = total_reranked_accuracy / len(EVAL_SET)
        print(f"Average Initial Retrieval Accuracy@1_any: {avg_initial:.2f}")
        print(f"Average Reranked Accuracy@1_any: {avg_reranked:.2f}")

    elif len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
        asyncio.run(search_and_rerank(query_text))
    else:
        print("Usage:")
        print("  python search.py \"<query_text>\"")
        print("  python search.py --eval")
        print("Example: python search.py \"how to replace a shower cartridge\"")
