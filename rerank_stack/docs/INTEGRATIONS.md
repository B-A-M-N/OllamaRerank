# Integration Recipes

## Simple HTTP client (Python)
```python
import httpx

payload = {
    "model": "B-A-M-N/qwen3-reranker-0.6b-fp16",
    "query": "car maintenance",
    "documents": [
        {"text": "Spark plug replacement for a Civic...", "original_index": 0},
        {"text": "Maintaining your car's engine oil...", "original_index": 1},
    ],
    "policy_id": "howto_automotive_v1",
}

resp = httpx.post("http://127.0.0.1:8000/api/rerank", json=payload, timeout=30.0)
resp.raise_for_status()
for r in resp.json()["results"]:
    print(r["original_corpus_index"], r["score"], r["facts"].get("decision"))
```

## LangChain post-retrieval reranker (sketch)
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import httpx

# after you get retriever results:
docs = retriever.get_relevant_documents("car maintenance")
payload = {
    "model": "B-A-M-N/qwen3-reranker-0.6b-fp16",
    "query": "car maintenance",
    "documents": [
        {"text": d.page_content, "original_index": i} for i, d in enumerate(docs)
    ],
    "policy_id": "howto_automotive_v1",
}
rerank = httpx.post("http://127.0.0.1:8000/api/rerank", json=payload, timeout=30.0).json()
ranked_docs = sorted(zip(docs, rerank["results"]), key=lambda x: x[1]["score"], reverse=True)
```

## LlamaIndex rerank postprocessor (sketch)
```python
from llama_index.postprocessor.types import BaseNodePostprocessor
import httpx

class RerankServicePostprocessor(BaseNodePostprocessor):
    def __init__(self, url: str, model: str, policy_id: str):
        self.url = url
        self.model = model
        self.policy_id = policy_id

    def postprocess_nodes(self, nodes, query_str=None, **kwargs):
        payload = {
            "model": self.model,
            "query": query_str,
            "documents": [
                {"text": n.get_content(), "original_index": idx} for idx, n in enumerate(nodes)
            ],
            "policy_id": self.policy_id,
        }
        resp = httpx.post(self.url, json=payload, timeout=30.0)
        resp.raise_for_status()
        scored = resp.json()["results"]
        by_idx = {r["index"]: r["score"] for r in scored}
        return sorted(nodes, key=lambda n: by_idx.get(n.index, 0), reverse=True)
```

Notes:
- Keep `policy_id` consistent with your config in `configs/`.
- Respect budgets/timeouts; use the env knobs in `docs/USAGE.md` for production.
- For public exposure, put the API behind your proxy/auth/TLS; it binds local-only by default.
