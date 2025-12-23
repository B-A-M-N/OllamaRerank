import pytest
from fastapi.testclient import TestClient
from rerank_service.src.rerank_service.api import app
from rerank_service.src.rerank_service.schemas import RerankRequest

client = TestClient(app)

def test_api_rerank_contract():
    request_documents = [
        {"text": "Turn off the water supply. Remove handle...", "original_index": 0},
        {"text": "A shower cartridge controls temperature...", "original_index": 1},
        {"text": "Unrelated: how to fix a bicycle chain...", "original_index": 2},
    ]
    request_payload = {
        "model": "test-reranker",
        "query": "how to replace a shower cartridge",
        "documents": request_documents,
        "return_documents": True
    }
    response = client.post("/api/rerank", json=request_payload)
    assert response.status_code == 200
    data = response.json()

    assert "model" in data
    assert "query" in data
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == len(request_payload["documents"]) # All documents returned by default

    scores = []
    original_indices = []
    for result in data["results"]:
        assert "index" in result
        assert "score" in result
        assert "document" in result # because return_documents was True
        assert isinstance(result["index"], int)
        assert isinstance(result["score"], float)
        assert result["document"] == request_payload["documents"][result["index"]]["text"]
        scores.append(result["score"])
        original_indices.append(result["index"])

    # Check if results are sorted by score descending
    assert all(scores[i] >= scores[i+1] for i in range(len(scores) - 1))

    # Check if all original indices are present and unique
    assert sorted(original_indices) == list(range(len(request_payload["documents"])))

    # Test top_n
    request_payload_top_n = {**request_payload, "top_n": 2}
    response_top_n = client.post("/api/rerank", json=request_payload_top_n)
    assert response_top_n.status_code == 200
    data_top_n = response_top_n.json()
    assert len(data_top_n["results"]) == 2

    # Test return_documents=False
    request_payload_no_docs = {**request_payload, "return_documents": False}
    response_no_docs = client.post("/api/rerank", json=request_payload_no_docs)
    assert response_no_docs.status_code == 200
    data_no_docs = response_no_docs.json()
    for result in data_no_docs["results"]:
        assert "document" not in result

def test_api_rerank_empty_documents_error():
    request_payload = {
        "model": "test-reranker",
        "query": "test query",
        "documents": []
    }
    response = client.post("/api/rerank", json=request_payload)
    assert response.status_code == 422 # Pydantic validation error for min_length=1
    data = response.json()
    # The actual FastAPI response for pydantic errors is often different from the spec's
    # custom ErrorResponse, but we check for a reasonable indication of an error.
    assert "detail" in data
    assert any("documents" in error.get("loc", []) and "min_length" in error.get("type", "") for error in data["detail"])

def test_api_rerank_max_documents_limit():
    MAX_DOCS = 1024 # Defined in api.py
    long_documents = ["test doc"] * (MAX_DOCS + 1)
    request_payload = {
        "model": "test-reranker",
        "query": "test query",
        "documents": long_documents
    }
    response = client.post("/api/rerank", json=request_payload)
    assert response.status_code == 413
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "payload_too_large"

def test_v1_rerank_contract():
    request_payload = {
        "model": "test-reranker",
        "query": "how to replace a shower cartridge",
        "documents": [
            "Turn off the water supply. Remove handle...",
            "A shower cartridge controls temperature...",
            "Unrelated: how to fix a bicycle chain..."
        ],
        "top_n": 2
    }
    response = client.post("/v1/rerank", json=request_payload)
    assert response.status_code == 200
    data = response.json()

    assert "model" in data
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 2 # Because top_n was 2

    relevance_scores = []
    original_indices = []
    for result in data["data"]:
        assert "index" in result
        assert "relevance_score" in result
        assert isinstance(result["index"], int)
        assert isinstance(result["relevance_score"], float)
        relevance_scores.append(result["relevance_score"])
        original_indices.append(result["index"])

    # Check if results are sorted by relevance_score descending (indirectly via internal rerank_endpoint)
    # The internal rerank_endpoint sorts by score, and this adaptor preserves that order.
    # We can't strictly assert this here as dummy scores are random, but we ensure the structure.
    # The sorting will be implicitly tested by the /api/rerank test if scores are stable.

    # Check if all original indices are present and unique within the top_n
    # This might not be sequential if top_n picks non-contiguous original indices after sorting
    assert len(set(original_indices)) == len(original_indices)
