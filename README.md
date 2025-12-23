# OllamaRerank

Precision-first reranking for embedding search. It takes your top-K candidates, gates out wrong domains/objects (shower vs faucet, car vs bike), scores what’s left with a lightweight reranker, and returns FACTS that explain every decision.

## What it does
- **Fixes embedding mistakes**: promotes the right item (e.g., shower cartridge) above wrong-but-similar items (e.g., kitchen faucet).
- **Hard gates**: domain/item mismatches get score 0 with a `zero_reason`.
- **Ambiguity handling**: single-word/generic queries (e.g., “maintenance”) skip rerank and fall back to similarity.
- **Explainable**: every result carries FACTS (domains, items, zero_reason, rerank_mode).

## Quickstart
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# start API (adjust module/path if needed)
PYTHONPATH=src uvicorn rerank_service.api:app --host 127.0.0.1 --port 8000 --reload

# run bundled demos
python scripts/run_demo_cases.py
```

## Pipeline in 5 bullets
1) **Retrieve** with embeddings (top-K).
2) **Gate**: skip rerank if query is ambiguous/short; otherwise apply domain/item gates.
3) **Score**: coarse buckets (0/1/2/3) for usefulness.
4) **Sort**: positives first; zeros ordered by reject tier (adjacent vs domain mismatch) then similarity.
5) **Explain**: FACTS show domains/items/zero_reason/rerank_mode.

## API (shape)
`POST /api/rerank`
```json
{
  "model": "B-A-M-N/qwen3-reranker-0.6b-fp16",
  "query": "how to fix a leaky shower",
  "documents": [
    {"text": "doc text", "original_index": 123}
  ],
  "policy_id": "howto_plumbing_v1"
}
```
Response: `results` with `score` + `facts`. `/v1/rerank` is an OpenAI-style convenience.

## Models
- Demo client defaults to `B-A-M-N/qwen3-reranker-0.6b-fp16` (Ollama).
- Set your own via request `model`, env var, or config. Server forwards the string to Ollama’s `/api/generate`.
- Add a `/meta`/`/health` endpoint to surface `{model, backend, device}` if you customize.

## Policies
- YAML files in `configs/` (e.g., `howto_plumbing_v1.yaml`, `howto_automotive_v1.yaml`).
- Define domains, item aliases, ambiguous triggers, optional caps/boosts, few-shot.
- `policy_id` maps directly to the filename (minus extension).

## Integrations (patterns)
- **Python/RAG**: retrieve → call `/api/rerank` → keep top-K reranked docs as context.
- **LangChain**: wrap rerank as a `DocumentCompressor`/post-processor.
- **LlamaIndex**: add as a `node_postprocessor`.
- **Haystack/other**: insert after BM25/Embedding retrievers; keep `(query, documents[]) -> scored docs`.

Minimal Python skeleton:
```python
import requests

def rerank(query, candidates):
    payload = {
        "query": query,
        "model": "B-A-M-N/qwen3-reranker-0.6b-fp16",
        "documents": [
            {"text": c["text"], "original_index": c["id"]} for c in candidates
        ],
    }
    r = requests.post("http://127.0.0.1:8000/api/rerank", json=payload, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])
```

## Adding your own data
```bash
# assumes corpus.json exists (list of texts)
python scripts/build_index.py \
  --model nomic-embed-text \
  --corpus-file corpus.json \
  --output-index index.npz \
  --output-docs docs.json

# add eval cases to scripts/run_demo_cases.py (EVAL_SET)
python scripts/run_demo_cases.py
```

## Docs & Troubleshooting
See `docs/USAGE.md` for:
- Full pipeline walkthrough
- Request/response schema
- Model/backends and hardware notes
- Policy config knobs
- Ambiguity handling
- FACTS/zero_reason meanings
- Troubleshooting and recipes

## License
MIT (see LICENSE).
