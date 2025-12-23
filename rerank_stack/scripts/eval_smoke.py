import httpx
import asyncio
import time
import random
from typing import List, Dict, Any, Tuple

# Assuming the API is running at this address
API_BASE_URL = "http://127.0.0.1:8000"

# --- Sample Test Data (very basic for a smoke test) ---
# Each tuple: (query, [(document, is_relevant), ...])
EVAL_DATA = [
    (
        "cat vaccination schedule",
        [
            ("Kittens need vaccines at 6-8 weeks, then boosters.", True),
            ("How to change spark plugs in a car.", False),
            ("Common cat diseases and prevention.", True),
            ("Dog training tips.", False),
        ],
    ),
    (
        "best way to clean a carpet",
        [
            ("Vacuum regularly and treat stains immediately with a cleaner.", True),
            ("How to bake a cake from scratch.", False),
            ("Professional carpet cleaning services.", True),
            ("Gardening tools and their uses.", False),
        ],
    ),
]

async def run_evaluation():
    print("Starting rerank API evaluation smoke test...")
    
    client = httpx.AsyncClient(base_url=API_BASE_URL)
    
    all_latencies = []
    correct_rankings = 0
    total_relevant_pairs = 0
    total_score_gaps = []

    for query, doc_sets in EVAL_DATA:
        documents = [
            {"text": doc_text, "original_index": i}
            for i, (doc_text, _) in enumerate(doc_sets)
        ]
        relevant_indices = {i for i, (doc_text, is_relevant) in enumerate(doc_sets) if is_relevant}

        request_payload = {
            "model": "llama2",  # Placeholder model name
            "query": query,
            "documents": documents,
            "return_documents": False # No need to return documents for evaluation metrics
        }

        start_time = time.time()
        try:
            response = await client.post("/api/rerank", json=request_payload)
            response.raise_for_status()
            end_time = time.time()
            all_latencies.append((end_time - start_time) * 1000) # Convert to ms
            
            data = response.json()
            results: List[Dict[str, Any]] = data.get("results", [])

            if not results:
                print(f"  Warning: No results for query '{query}'")
                continue

            # --- Evaluate Accuracy (simple: is the top result relevant?) ---
            if results[0]["original_corpus_index"] in relevant_indices:
                correct_rankings += 1

            # --- Mean Score Gap between relevant/irrelevant ---
            relevant_scores = [r["score"] for r in results if r["original_corpus_index"] in relevant_indices]
            irrelevant_scores = [r["score"] for r in results if r["original_corpus_index"] not in relevant_indices]
            
            if relevant_scores and irrelevant_scores:
                # Simple average of all relevant vs all irrelevant
                mean_relevant_score = sum(relevant_scores) / len(relevant_scores)
                mean_irrelevant_score = sum(irrelevant_scores) / len(irrelevant_scores)
                total_score_gaps.append(mean_relevant_score - mean_irrelevant_score)
            
            total_relevant_pairs += len(relevant_indices)


        except httpx.HTTPStatusError as e:
            print(f"  API Error for query '{query}': {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Rerank API error: {e}") from e
        except httpx.RequestError as e:
            print(f"  Request Error for query '{query}': {e}")
            raise RuntimeError(f"Rerank API request failed: {e}") from e
        except Exception as e:
            print(f"  Unexpected Error for query '{query}': {e}")

    await client.aclose()

    print("\n--- Evaluation Results ---")
    if EVAL_DATA:
        print(f"Number of test queries: {len(EVAL_DATA)}")
        
        # Accuracy @1 (simple)
        accuracy_at_1 = (correct_rankings / len(EVAL_DATA)) if EVAL_DATA else 0
        print(f"Accuracy @1 (Top result relevant): {accuracy_at_1:.2f}")

        # Mean Score Gap
        if total_score_gaps:
            avg_score_gap = sum(total_score_gaps) / len(total_score_gaps)
            print(f"Mean (relevant - irrelevant) score gap: {avg_score_gap:.2f}")
        else:
            print("Could not calculate mean score gap (not enough relevant/irrelevant pairs).")

        # Latency Stats
        if all_latencies:
            avg_latency = sum(all_latencies) / len(all_latencies)
            min_latency = min(all_latencies)
            max_latency = max(all_latencies)
            print(f"Average latency per request: {avg_latency:.2f} ms")
            print(f"Min latency: {min_latency:.2f} ms, Max latency: {max_latency:.2f} ms")
        else:
            print("No latency data collected.")
    else:
        print("No evaluation data to process.")

if __name__ == "__main__":
    # To run this script, ensure your FastAPI app is running in another terminal
    # e.g., PYTHONPATH=src uvicorn rerank_service.src.rerank_service.api:app --reload
    print(f"Connecting to API at {API_BASE_URL}")
    asyncio.run(run_evaluation())
