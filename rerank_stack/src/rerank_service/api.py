from fastapi import FastAPI, HTTPException, status
from typing import List, Dict, Any, Optional
import time
import random
import traceback

from .schemas import (
    RerankRequest,
    RerankResponse,
    Result,
    Usage,
    Timing,
    ErrorResponse,
    ErrorDetail,
    OpenAIRerankRequest,
    OpenAIRerankResponse,
    OpenAIRerankResult,
    OpenAIUsage,
    TruncateOptions,
    Options,
    Document,  # Import Document schema
)
from .truncate import truncate_text  # Import the truncation utility
from .ollama_client import OllamaClient  # Import OllamaClient
from .scoring import score_documents_concurrently # Import the renamed scoring function
from .doc_quality import rerank_postprocess # Import the new post-processor
from .policy import policy_manager, build_policy_context, PolicyContext # Re-introduce policy imports

app = FastAPI(
    title="Ollama Rerank API",
    description="A clean, first-class rerank API spec for Ollama.",
    version="0.1.0",
)

# Instantiate OllamaClient globally or use dependency injection
# For simplicity, we'll instantiate it here.
ollama_client = OllamaClient()

@app.on_event("shutdown")
async def shutdown_event():
    await ollama_client.client.aclose()


@app.get("/health")
async def health():
    return {"status": "ok"}

def clamp_bucket(score: int, facts: dict) -> int:
    # Hard domain gate
    if not facts.get("domain_match", False):
        return 0

    # Item identity gate: if query names an item and doc does NOT match it, clamp to lowest bucket (0)
    if facts.get("query_item") is not None and not facts.get("same_item", False):
        return 0

    # same item: only 2 or 3 allowed
    return 3 if score >= 3 else 2

@app.post(
    "/api/rerank",
    response_model=RerankResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
    },
)
async def rerank_endpoint(request: RerankRequest):
    start_total_time = time.time()

    print("\n--- New Rerank Request ---")
    print(f"Model: {request.model}")
    print(f"Query: {request.query}")
    print(f"Policy ID (request): {request.policy_id}")
    print("Documents (head):")
    for i, doc in enumerate(request.documents):
        print(f"  {i}: {doc.text[:100]}...")
    print("------------------------\n")

    # Placeholder for max documents limit
    MAX_DOCUMENTS = 1024
    if len(request.documents) > MAX_DOCUMENTS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="payload_too_large",
                    message=f"Too many documents. Max is {MAX_DOCUMENTS}",
                )
            ).model_dump(),
        )

    load_ms = 22 # Dummy value

    # Apply truncation
    truncate_opts = request.truncate or TruncateOptions()
    truncated_query = truncate_text(request.query, truncate_opts.query_tokens)
    
    # Adapt to new documents schema (List[Document])
    documents_for_scoring = []
    for request_index, doc_req_item in enumerate(request.documents):
        truncated_text = truncate_text(doc_req_item.text, truncate_opts.document_tokens)
        documents_for_scoring.append(
            (request_index, doc_req_item.original_index, truncated_text, len(doc_req_item.text))
        )


    # Update dummy token counts to reflect character length after truncation
    query_tokens = len(truncated_query)
    document_tokens_list = [len(doc_text) for _, _, doc_text, _ in documents_for_scoring]
    total_document_tokens = sum(document_tokens_list)


    start_encode_time = time.time()
    # In a real scenario, documents and query would be encoded here,
    # potentially part of the Ollama call. For now, this is dummy.
    encode_ms = int((time.time() - start_encode_time) * 1000)

    start_score_time = time.time()
    policy = policy_manager.select_policy(request.policy_id, request.query)
    print(f"[RERANK_REQ] query={repr(request.query)} policy={policy.name} (default={policy_manager.default_policy_id})")
    context = build_policy_context(policy)
    
    # Pre-compute identity facts for all documents once.
    original_docs_map = {doc.original_index: doc.text for doc in request.documents}
    facts_map = {
        orig_idx: context.compute_identity_facts(query=request.query, doc=original_docs_map.get(orig_idx, ""))
        for _, orig_idx, _, _ in documents_for_scoring
    }
    
    # This now returns a tuple: (scored_documents, warnings)
    llm_scored_documents, warnings = await score_documents_concurrently(
        ollama_client,
        request.model,
        truncated_query,
        documents_for_scoring,
        request.options or Options(),
        context,
        facts_map, # Pass pre-computed facts
    )

    # --- Sanity check for missing documents ---
    expected_indices = {doc.original_index for doc in request.documents}
    returned_indices = {orig_idx for _, orig_idx, _, _ in llm_scored_documents}
    if expected_indices - returned_indices:
        print(f"DEBUG: MISSING documents from scoring response: {sorted(expected_indices - returned_indices)}")
    # -----------------------------------------

    # --- Clamp scores based on FACTS ---
    clamped_scored_documents = []
    for req_idx, orig_idx, score, tie_breaker in llm_scored_documents:
        facts = facts_map.get(orig_idx, {})
        new_score = clamp_bucket(score, facts)
        if new_score != score:
            print(f"### ITEM_CLAMP_ACTIVE ### Query: {request.query}, Doc: {orig_idx}. Reason: FACTS based clamp. Score: {score} -> {new_score}")
        clamped_scored_documents.append((req_idx, orig_idx, new_score, tie_breaker))
    
    llm_scored_documents = clamped_scored_documents
    # --- End of clamp ---

    # --- Post-process and apply quality heuristics ---
    
    # llm_scored_documents is a list of 4-element tuples from the successful scores
    docs_for_postprocess = [original_docs_map[original_corpus_index] for _, original_corpus_index, _, _ in llm_scored_documents]
    scores_for_postprocess = [score for _, _, score, _ in llm_scored_documents]
    max_rank = max(1, len(llm_scored_documents) - 1)
    original_similarities = [
        1.0 - (req_idx / max_rank) for req_idx, _, _, _ in llm_scored_documents
    ]
    
    postprocessed_results = rerank_postprocess(
        query=request.query,
        docs=docs_for_postprocess,
        rerank_scores=scores_for_postprocess,
        original_similarities=original_similarities,
        context=context # Pass the context to rerank_postprocess
    )
    
    # --- Final result construction ---
    ranked_results: List[Result] = []
    for relative_idx, final_score, _ in postprocessed_results:
        # The `relative_idx` from postprocessing corresponds to the original request order
        # of the documents *that were scored* (not including gate-skipped ones).
        # We need to look up the original corpus index from our `llm_scored_documents` list.
        _req_idx, original_corpus_index, _score, _tie_breaker = llm_scored_documents[relative_idx]

        facts = facts_map.get(original_corpus_index, {}).copy() if facts_map.get(original_corpus_index) else {}
        if final_score == 0:
            if not facts.get("same_domain", False):
                facts["zero_reason"] = "domain_mismatch"
            elif not facts.get("doc_items"):
                facts["zero_reason"] = "no_item_found"
            elif facts.get("query_item") is not None and facts.get("same_item") is False:
                facts["zero_reason"] = "item_mismatch"
            else:
                facts["zero_reason"] = "low_signal"

        result = Result(
            index=relative_idx, # The final rank position
            original_corpus_index=original_corpus_index,
            score=final_score,
            facts=facts
        )
        if request.return_documents:
            result.document = original_docs_map.get(original_corpus_index)
        ranked_results.append(result)

    # Partition: keep positive scores ahead of zeros for clearer ordering.
    positives = [r for r in ranked_results if (r.score or 0) > 0]
    zeros = [r for r in ranked_results if (r.score or 0) <= 0]
    ranked_results = positives + zeros

    # Handle top_n
    if request.top_n is not None and request.top_n > 0:
        ranked_results = ranked_results[: request.top_n]

    score_ms = int((time.time() - start_score_time) * 1000)
    total_ms = int((time.time() - start_total_time) * 1000)

    usage = Usage(
        pairs=len(request.documents),
        query_tokens=query_tokens,
        document_tokens=total_document_tokens,
        total_tokens=query_tokens + total_document_tokens,
    )
    timing = Timing(
        load_ms=load_ms,
        encode_ms=encode_ms,
        score_ms=score_ms,
        total_ms=total_ms,
    )

    return RerankResponse(
        model=request.model,
        query=request.query,
        results=ranked_results,
        usage=usage,
        timing=timing,
        warnings=warnings,
    )


@app.post(
    "/v1/rerank",
    response_model=OpenAIRerankResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
    },
)
async def openai_rerank_endpoint(request: OpenAIRerankRequest):
    # Convert OpenAI-ish request to internal RerankRequest
    # NOTE: OpenAIRerankRequest still has documents: List[str]
    # We need to create dummy Document objects for the internal request.
    internal_documents = [
        Document(text=doc_text, original_index=i) # Assign a dummy original_index
        for i, doc_text in enumerate(request.documents)
    ]

    internal_request = RerankRequest(
        model=request.model,
        query=request.query,
        documents=internal_documents, # Pass the new Document objects
        top_n=request.top_n,
        return_documents=False, # OpenAI-ish response doesn't include documents in the top-level data
        truncate=None,
        batch_size=16,
        options=None,
    )

    # Call the core reranking logic
    internal_response = await rerank_endpoint(internal_request)

    # Convert internal RerankResponse to OpenAI-ish response
    openai_results = [
        OpenAIRerankResult(
            index=r.index,
            relevance_score=r.score,
            original_corpus_index=r.original_corpus_index
        )
        for r in internal_response.results
    ]

    openai_usage = None
    if internal_response.usage:
        openai_usage = OpenAIUsage(total_tokens=internal_response.usage.total_tokens)

    return OpenAIRerankResponse(
        model=internal_response.model, data=openai_results, usage=openai_usage
    )
