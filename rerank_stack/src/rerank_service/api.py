import sys
import sys
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import time
import random
import traceback
import os
from pprint import pprint

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
from .scoring import score_documents_concurrently, fine_score_documents # Import the renamed scoring function
from .parser import fallback_fine_score
from .policies import load_policy

# Safety: warn if binding to public while RERANK_BIND_PUBLIC=0 (default)
def _warn_public_bind():
    bind_public_env = os.getenv("RERANK_BIND_PUBLIC", "0").lower() not in {"0", "false", "no"}
    bind_host = os.getenv("HOST", "")
    uvicorn_host = os.getenv("UVICORN_HOST", "")
    cli_host = ""
    # Detect uvicorn CLI host from environment if set; otherwise rely on HOST/UVICORN_HOST
    for var in ("UVICORN_HOST", "HOST"):
        val = os.getenv(var)
        if val:
            cli_host = val
            break
    target_host = cli_host or bind_host or ""
    if (target_host == "0.0.0.0") and not bind_public_env:
        print(
            "[SECURITY WARNING] RERANK_BIND_PUBLIC=0 and host=0.0.0.0. "
            "Binding to all interfaces without auth/TLS is unsafe. "
            "Set RERANK_BIND_PUBLIC=1 to suppress this warning."
        )
        sys.exit(1)

_warn_public_bind()

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

@app.exception_handler(Exception)
async def all_exception_handler(request, exc):
    tb = traceback.format_exc()
    print(tb)
    return JSONResponse(status_code=500, content={"error": "internal_error", "traceback": tb})


@app.get("/health")
async def health():
    return {"status": "ok"}

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
            (request_index, doc_req_item.original_index, truncated_text, len(doc_req_item.text), doc_req_item.similarity)
        )


    # Update dummy token counts to reflect character length after truncation
    query_tokens = len(truncated_query)
    document_tokens_list = [len(doc_text) for _, _, doc_text, _, _ in documents_for_scoring]
    total_document_tokens = sum(document_tokens_list)


    start_encode_time = time.time()
    # In a real scenario, documents and query would be encoded here,
    # potentially part of the Ollama call. For now, this is dummy.
    encode_ms = int((time.time() - start_encode_time) * 1000)

    start_score_time = time.time()
    # Policy selection (legacy/demo preserved via policy plugin)
    policy_override = request.policy_id or os.environ.get("RERANK_POLICY_ID")
    policy = load_policy(policy_override or "legacy_demo")
    context = policy.build_context(request.query, policy_override)
    print(f"[RERANK_REQ] query={repr(request.query)} policy={policy.policy_id()}")
    
    # Pre-compute identity facts for all documents once.
    facts_map = policy.precompute_facts(request.query, request.documents)
    print("FACTS_MAP_SAMPLE:", list(facts_map.items())[:2])
    
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

    similarities_by_req: List[Optional[float]] = [None] * len(request.documents)
    for req_idx, _, _, _, sim in documents_for_scoring:
        similarities_by_req[req_idx] = sim

    debug = bool(request.options.debug) if request.options else False
    ranked_results = policy.rank(
        query=request.query,
        documents=request.documents,
        scored_docs=llm_scored_documents,
        similarities_by_req=similarities_by_req,
        facts_map=facts_map,
        debug=debug,
        options=request.options or Options(),
    )
    for res in ranked_results[:2]:
        print(
            "FACTS_FOR_DOC",
            res.original_corpus_index,
            facts_map.get(res.original_corpus_index),
            "result_facts",
            res.facts,
        )

    # Fine-grained tie-breaker: only for ACCEPT docs, top-K
    TOP_K_FINE = int(os.getenv("RERANK_FINE_TOP_K", "10"))
    doc_text_by_orig = {doc.original_index: doc.text for doc in request.documents}
    sim_by_orig = {
        doc.original_index: (float(doc.similarity) if doc.similarity is not None else 0.0)
        for doc in request.documents
    }

    accept_candidates = [r for r in ranked_results if (r.decision or "").upper() == "ACCEPT"]
    accept_for_fine = accept_candidates[:TOP_K_FINE] if TOP_K_FINE > 0 else []
    fine_scores: Dict[int, Dict[str, Any]] = {}

    if accept_for_fine:
        fine_inputs = []
        for res in accept_for_fine:
            doc_text = doc_text_by_orig.get(res.original_corpus_index, "")
            if doc_text:
                fine_inputs.append((res.original_corpus_index, doc_text))
        fine_scores = await fine_score_documents(
            ollama_client=ollama_client,
            model=request.model,
            query=request.query,
            docs=fine_inputs,
        )

    for res in ranked_results:
        meta = fine_scores.get(res.original_corpus_index)
        res.tie_breaker_model = request.model
        res.tie_breaker_prompt_id = "percent_line_v1"
        res.tie_breaker_used = False
        res.tie_breaker_error = None
        res.tie_breaker_latency_ms = None
        res.fine_score = None
        res.fine_score_norm = None
        fallback_used = False
        if meta:
            res.tie_breaker_latency_ms = meta.get("latency_ms")
            meta_error = meta.get("error")
            if meta_error:
                res.tie_breaker_error = meta_error
            if meta.get("fine_score") is not None:
                res.fine_score = meta["fine_score"]
                res.fine_score_norm = meta.get("fine_score_norm")
                res.tie_breaker_used = True
            else:
                # Deterministic fallback when tie-break fails
                intent_fit = (res.facts or {}).get("intent_fit") if res.facts else None
                sim_val = sim_by_orig.get(res.original_corpus_index, 0.0)
                fallback_score = fallback_fine_score(intent_fit, sim_val)
                res.fine_score = fallback_score
                res.fine_score_norm = fallback_score / 100.0
                fallback_used = True
                res.tie_breaker_used = True
                if meta_error:
                    res.tie_breaker_error = f"{meta_error}|fallback"
        if res.facts is None:
            res.facts = {}
        res.facts.update(
            {
                "fine_score": res.fine_score,
                "fine_score_norm": res.fine_score_norm,
                "tie_breaker_used": res.tie_breaker_used,
                "tie_breaker_model": res.tie_breaker_model,
                "tie_breaker_prompt_id": res.tie_breaker_prompt_id,
                "tie_breaker_error": res.tie_breaker_error,
                "tie_breaker_latency_ms": res.tie_breaker_latency_ms,
                "tie_breaker_fallback": fallback_used,
            }
        )

    # Final deterministic sort with tie-breaker fields
    def _bucket(decision: Optional[str]) -> int:
        if not decision:
            return 0
        d = decision.upper()
        if d == "ACCEPT":
            return 2
        if d == "KEEP_LOW_SIGNAL":
            return 1
        return 0

    ranked_results.sort(
        key=lambda r: (
            -_bucket(r.decision),
            -(r.tier if r.tier is not None else -1),
            -(r.fine_score if r.fine_score is not None else -1),
            -(r.rank_score if r.rank_score is not None else (r.score or 0)),
            -sim_by_orig.get(r.original_corpus_index, 0.0),
            r.original_corpus_index,  # deterministic tie-of-ties
            r.index,
        )
    )

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

    response_payload = RerankResponse(
        model=request.model,
        query=request.query,
        results=ranked_results,
        usage=usage,
        timing=timing,
        warnings=warnings,
    )

    # Belt-and-suspenders: ensure facts is always a dict for every result.
    for res in response_payload.results:
        if res.facts is None:
            res.facts = {}

    if debug:
        print("DEBUG outgoing payload:")
        pprint(response_payload.model_dump())

    return response_payload


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
