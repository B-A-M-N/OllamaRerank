# OllamaRerank

Precision-first reranking for embedding search. This stack takes your top‑K vector candidates, applies domain/item gates, scores what’s left with a lightweight reranker, and returns **FACTS** explaining every decision. The goal: fix “right domain, wrong object” mistakes (shower vs faucet, car vs bike) without heavy compute.

## Why this exists
Embedding retrieval is great at “nearby semantics,” but bad at **object identity**. Reranking fixes that:
- **Correct object wins** (shower cartridge beats kitchen faucet).
- **Wrong domain is rejected** (bicycle docs drop for car queries).
- **Explainable output** (every score has a reason).

## Value proposition
- Fix “right domain, wrong object” errors without heavy compute.
- Provide deterministic reasons for rejection (item/domain mismatch).
- Keep retrieval fast while improving top‑1/top‑K relevance.

## What it does
- **Hard gates**: item/domain mismatches become score `0` with `zero_reason`.
- **Coarse scoring**: small buckets (0/1/2/3) for usefulness.
- **Deterministic ordering**: accepted docs first; rejects tiered by reason, then similarity (tie-of-ties stable).
- **Ambiguity fallback**: vague/short queries skip the rerank API and fall back to similarity with lightweight intent‑aware heuristics.

## Quickstart
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# start API (local-only safe default)
PYTHONPATH=src uvicorn rerank_service.api:app --host 127.0.0.1 --port 8000 --reload

# run bundled demos
python scripts/run_demo_cases.py
```

## Eval
```bash
PYTHONPATH=src python rerank_demo/search.py --eval
```
See `docs/EVAL.md` to add your own queries.

## Docker
```bash
docker compose up --build
# API at http://127.0.0.1:8000 (binds local by default via RERANK_BIND_PUBLIC guard)
```

## How it works (pipeline)
1. **Retrieve**: embedding similarity fetches top‑K candidates.
2. **Ambiguity gate**: short/generic queries skip rerank.
3. **Rerank**: `/api/rerank` scores candidates.
4. **Gates**: item/domain mismatch ⇒ score 0 + `zero_reason`.
5. **Sort**: positives first; rejects tiered, then similarity; stable tie-of-ties.
6. **Explain**: FACTS show item/domain decisions and reasons.

## Rerank API (shape)
`POST /api/rerank`
```json
{
  "model": "B-A-M-N/qwen3-reranker-0.6b-fp16",
  "query": "how to fix a leaky shower",
  "documents": [
    {"text": "doc text", "original_index": 123, "similarity": 0.62}
  ],
  "policy_id": "howto_plumbing_v1"
}
```
Response includes `results` with `score` + optional `facts`. `/v1/rerank` is an OpenAI‑style convenience that accepts `documents: List[str]`.

## Model (default)
- Demo model: `B-A-M-N/qwen3-reranker-0.6b-fp16` (Ollama).
- The server forwards `model` directly to Ollama `/api/generate`.
- Swap models by changing `model` in the request or `RERANK_MODEL` in the environment; keep the score contract in sync.

## Policies
- Configs live in `configs/*.yaml`.
- `policy_id` maps directly to filenames in `configs/` (e.g., `howto_plumbing_v1` → `configs/howto_plumbing_v1.yaml`).
- Policies define domain anchors, item aliases, ambiguity triggers, and optional few‑shot examples.

## Integrations (pattern)
Retrieve → call `/api/rerank` → keep top‑K reranked docs as context.
```python
import requests

def rerank(query, candidates):
    payload = {
        "query": query,
        "model": "B-A-M-N/qwen3-reranker-0.6b-fp16",
        "documents": [
            {"text": c["text"], "original_index": c["id"], "similarity": c["similarity"]}
            for c in candidates
        ],
    }
    r = requests.post("http://127.0.0.1:8000/api/rerank", json=payload, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])
```

## Custom data (quick example)
```bash
# build an embedding index from corpus.json
python scripts/build_index.py \
  --model nomic-embed-text \
  --corpus-file corpus.json \
  --output-index index.npz \
  --output-docs docs.json

# run demos / eval
python scripts/run_demo_cases.py
```

## Eval results (bundled sample)
- Dataset: bundled demo (shower, car, bicycle, faucet, maintenance)
- Metrics (Accuracy@1 / MRR): initial ≈ 0.60 / 0.80 → reranked = 1.00 / 1.00  
Reproduce: `PYTHONPATH=src python rerank_demo/search.py --eval`

## Docs
See `docs/USAGE.md` for pipeline/API/ops, `docs/POLICY_SETUP.md` for policy tuning, `docs/EVAL.md` for eval, and `docs/INTEGRATIONS.md` for adapters.
