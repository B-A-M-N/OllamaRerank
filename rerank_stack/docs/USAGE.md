# OllamaRerank — How to Use Guide

This rerank stack fixes the classic “embedding pulled the wrong object” problem (e.g., faucet vs shower). It retrieves with embeddings, gates by domain/item, scores with a tiny reranker, and returns FACTS explaining every decision.

---

## Quickstart

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# start the rerank API (adjust module if needed)
PYTHONPATH=src uvicorn rerank_service.api:app --host 127.0.0.1 --port 8000 --reload

# run the bundled demos
python scripts/run_demo_cases.py
```

You should see initial retrieval, reranked results (with FACTS), and eval metrics (Accuracy@1, MRR, Recall@3_any/all).

---

## Model used

The rerank API forwards your `model` string to Ollama `/api/generate`.

Default demo model:
- `B-A-M-N/qwen3-reranker-0.6b-fp16`

To switch models:
1) `ollama pull <model-name>`
2) pass `model` in the API request or set `RERANK_MODEL` in the server environment.

If you’re unsure which model is active, add a `/meta` endpoint or log the model name on server startup.

---

## Pipeline (mental model)

1) **Initial retrieval**: embedding similarity fetches top-K candidates.  
2) **Ambiguity gate**: single-word/generic queries (e.g., “maintenance”, “help”) skip rerank and fall back to similarity.  
3) **Rerank API**: `/api/rerank` scores the remaining candidates.  
4) **Hard gates**: domain mismatch or item mismatch ⇒ score=0 with `zero_reason`.  
5) **Scoring**: coarse buckets (0/1/2/3) for usefulness.  
6) **Sort**: positives first; zeros ordered by reject tier, then similarity.  
7) **Facts**: every result carries domains, items, and `zero_reason`.  
8) **Eval**: Accuracy@1, MRR, Recall@3_any/all.

---

## What “good” looks like (shower demo)

Initial retrieval (similarity) top-1 is wrong:
* “How to fix a leaky faucet…” (similarity 0.7266)

Rerank fixes it:
* Rank 1: “Shower cartridge replacement…” (score 3.0)

Metrics jump:
* Accuracy@1_any: 0 → 1
* MRR: 0.50 → 1.00

---

## Calling the API yourself

### Request (template)
```json
POST /api/rerank
{
  "model": "B-A-M-N/qwen3-reranker-0.6b-fp16",
  "query": "how to fix a leaky shower",
  "documents": [
    {"text": "How to fix a leaky faucet in the kitchen.", "original_index": 4},
    {"text": "Shower cartridge replacement: turn off water...", "original_index": 0}
  ],
  "policy_id": "howto_plumbing_v1"
}
```

`original_index` is required so scores map back to your corpus.

### Response (shape)
```
results: [
  {
    original_corpus_index: 0,
    score: 3.0,
    facts: {
      query_item: "shower",
      doc_items: ["shower"],
      same_item: true,
      query_domain: "plumbing",
      doc_domain: "plumbing",
      zero_reason: null
    }
  },
  ...
]
```

Scores are coarse buckets: 3 (best), 2 (relevant/diagnosis), 1 (same domain but wrong/absent item), 0 (rejected).

### OpenAI-style endpoint
`/v1/rerank` accepts `documents: List[str]` and returns `relevance_score` (same buckets).

---

## FACTS fields (how to read them)

* `query_item`, `doc_items` — extracted objects (e.g., shower, faucet, car, bicycle).
* `same_item` — True/False/None (None means no item found in doc).
* `query_domain`, `doc_domain` — inferred domain (plumbing/automotive/bicycle/None).
* `same_domain` — whether domains match.
* `zero_reason` — why a doc was rejected:
  * `domain_mismatch`
  * `item_mismatch` (right domain, wrong object)
  * `no_item_found` (domain OK, no object extracted)
  * `low_signal`
* `rerank_mode` — `rerank_api` or `fallback_similarity` (for skipped ambiguous queries).

---

## Ambiguity handling

Skip rerank and use similarity fallback when:
* No tokens
* Single-word queries (too_short)
* Generic triggers: `maintenance`, `help`, `issue`, `problem`

Log message:  
`Skipping rerank API: ambiguous/short query (reason). Using similarity fallback.`

Fallback score: `0.85 * similarity + 0.15 * token_overlap`

---

## Sorting and reject tiers

Final ordering:
1) `score > 0` sorted by (score desc, similarity desc)
2) `score == 0` sorted by:
   * reject tier: same-domain adjacent (`no_item_found`/`item_mismatch`) before domain mismatches
   * then similarity desc

This makes the “accepted” vs “rejected” split obvious; zeros are still similarity-ordered within their tier.

---

## Config knobs (policies)

Policies live in `configs/*.yaml`:
* **Domains**: anchor words for plumbing/automotive/bicycle
* **Items**: aliases for objects (shower, faucet, car, bicycle, etc.)
* **Caps/boosts**: optional rubric adjustments (most gating is now deterministic)
* **Ambiguous triggers**: terms that force fallback
* **Few-shot**: optional examples per policy

Policy selection is automatic by query domain (e.g., `howto_plumbing_v1`), or override with `policy_id`.
**Policy IDs map 1:1 to filenames in `configs/`** (e.g., `howto_plumbing_v1` → `configs/howto_plumbing_v1.yaml`).

---

## Adding your own data

1) Put your corpus in `corpus.json` (list of texts).
   Example `corpus.json`:
   ```json
   [
     "Shower cartridge replacement: turn off water, remove handle, pull cartridge...",
     "Bicycle chain maintenance: clean, lube, check stretch with gauge...",
     "Spark plug replacement for a 2012 Civic: remove coil packs, torque to spec..."
   ]
   ```
2) Build embeddings (`index.npz`) via `scripts/build_index.py`. Example:
   ```bash
   # assumes corpus.json exists in repo root
   python scripts/build_index.py \
     --model nomic-embed-text \
     --corpus-file corpus.json \
     --output-index index.npz \
     --output-docs docs.json
   ```
   Notes:
   - `--corpus-file` should be a JSON list of strings (one doc per entry).
   - `--output-index` is the vector index used by the demo retrieval.
   - `--output-docs` is the aligned doc text file used by the demo runner.
   This writes `index.npz` + `docs.json` used by the demo/eval scripts.
3) Add eval cases to `scripts/run_demo_cases.py` (`EVAL_SET`).
4) Run `python scripts/run_demo_cases.py` and watch:
   * Accuracy@1_any
   * MRR
   * Recall@3_any / Recall@3_all

---

## Troubleshooting

* **All zeros / FLATLINE**: query is ambiguous or item/domain extraction failed. Check FACTS (`zero_reason`), and widen item aliases or domain anchors.
* **Wrong domain**: extend domain anchors in policy configs.
* **Item not found**: add aliases under `item_gating_rules` (e.g., faucet: tap/sink).
* **Rerank API errors**: ensure server is running, model exists locally, and `original_index` is provided.
* **Too strict**: relax hard clamp in `rerank_service.api.clamp_bucket` or allow same-domain/any-item to get bucket 1.

---

## Design choices (why)

* **Hard gates**: precision-first; block wrong domains/objects.
* **Ambiguity skip**: avoid wasting latency/LLM calls on “maintenance/help” with no anchors.
* **Coarse buckets + FACTS**: auditable decisions (zero_reason, domains, items).
* **Similarity tiebreaks**: stable ordering when scores tie.

---

## Recipe book (common tweaks)

---

## Glossary (quick reference)

* **FACTS**: per-document debug fields (item/domain decisions, zero_reason, etc.).
* **Coarse buckets**: the discrete score set used by the reranker (0/1/2/3).
* **Hard gates**: deterministic rejections (domain/item mismatch) that force score 0.
* **Ambiguous trigger**: a query with no item/domain anchors that skips the LLM rerank step.
* **Reject tiers**: ordered reasons for zero‑score docs (item_mismatch → no_item_found → domain_mismatch).

* More recall on specific queries → allow same-domain/any-item to score 1 instead of 0.
* Fewer skips → lower the ambiguity threshold (but expect noisier reranks).
* New domain → add anchors + item aliases in a new policy YAML; PolicyManager picks it up.
* Better tie handling → widen the score ladder or add more procedural boosts.

---

## Glossary (quick reference)

* **FACTS** — per-document debug fields (domains, items, zero_reason, rerank_mode) explaining why a score was assigned.
* **Coarse buckets** — small discrete scores (currently 0/1/2/3) used for reranking.
* **Hard gates** — deterministic rules that force score=0 for domain/item mismatches, regardless of model output.

---

## Model and backend

The reranker model is whatever you send in the request (`model` field). The demo client defaults to:

```
B-A-M-N/qwen3-reranker-0.6b-fp16
```

If you use Ollama:
1) Ensure the model is pulled: `ollama list` (pull if needed).
2) The server forwards the `model` string to Ollama’s `/api/generate`.
3) On each request the server logs `Model: <name>` — check logs to confirm the active model.

Optional: add a meta/health endpoint that returns `{model, backend, device}` so integrations can verify what’s running.

### Where to set the model

Pick one pattern and stick to it:
- **Env var** (recommended): `export RERANK_MODEL=B-A-M-N/qwen3-reranker-0.6b-fp16`
- Config file (YAML/JSON) that your server reads
- Hardcode in client/server (least ideal)

On startup, print the active model/backend/device. If it doesn’t, add a log. Clients should be able to hit `/health` or `/meta` and see:

```json
{
  "service": "rerank_api",
  "model": "B-A-M-N/qwen3-reranker-0.6b-fp16",
  "backend": "ollama",
  "device": "cpu|cuda",
  "version": "0.1.0"
}
```

### Running the model

**Ollama (local)**
```bash
ollama pull B-A-M-N/qwen3-reranker-0.6b-fp16
PYTHONPATH=src uvicorn rerank_service.api:app --host 127.0.0.1 --port 8000
```

**Swap in your own reranker**
- Any model that can score (query, doc) pairs and emit a small discrete score is fine (bge-reranker, cross-encoders, etc.). Keep the response contract (score + optional facts).

### Score meaning (current buckets)

* `3` = direct answer / actionable fix steps
* `2` = relevant / diagnosis / partial help
* `1` = same domain but wrong/absent item (only if you allow it)
* `0` = reject (domain mismatch, item mismatch, no signal)

If you later switch to `{0,3,7,10}`, document that mapping and update parsers/clients.

---

## API schema (generic)

`POST /api/rerank`

```json
{
  "model": "your-model-name",
  "query": "your query",
  "documents": [
    {"text": "doc text", "original_index": 123}
  ],
  "policy_id": "howto_plumbing_v1"
}
```

Response (shape):

```json
{
  "results": [
    {
      "original_corpus_index": 123,
      "score": 3.0,
      "facts": {...}
    }
  ]
}
```

`/v1/rerank` is an OpenAI-style convenience that accepts `documents: List[str]` and returns `relevance_score`.

---

## Integration patterns

### Plain Python / RAG
1) Retrieve top N via vector search.
2) Send to `/api/rerank`.
3) Use top K reranked docs as context.

### LangChain
* Wrap rerank API as a `DocumentCompressor`/post-processor after your retriever.

### LlamaIndex
* Add rerank as a `node_postprocessor` on retrieved nodes.

### Haystack / other
* Insert rerank after BM25/Embedding retrievers; keep schema: `(query, documents[]) -> scored docs`.

Always retrieve more than you keep (e.g., retrieve 30–50, keep top 5–10 after rerank).

**20-line Python skeleton**
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
    results = r.json().get("results", [])
    return [(res["original_corpus_index"], res["score"], res.get("facts")) for res in results]
```

---

## Deployment notes

* Local dev: Ollama + API server on one box.
* LAN/cluster: expose `/api/rerank` to other machines; set timeouts and max docs per call.
* Docker: containerize the API; mount or run Ollama alongside; optional Redis for caching.
* Production hygiene:
  * cache rerank responses for repeated queries,
  * cap doc length and doc count per call,
  * log p50/p95 latency and reject rates,
  * keep a `/health` endpoint.

### Hardware expectations

* Small rerankers (0.6B–4B) run fine on CPU for light loads; GPU recommended for higher throughput/latency SLOs.
* If you swap to larger models, expect to need a GPU and increase timeouts.

### Deployment checklist

* [ ] Model name + version/tag documented
* [ ] Backend (Ollama/llama.cpp/vLLM) documented
* [ ] Hardware noted (CPU/GPU)
* [ ] API schema documented (request/response)
* [ ] Scoring rubric documented (what 0/1/2/3 mean)
* [ ] Integration path documented (Python/LangChain/LlamaIndex)
* [ ] Failure modes documented (all 0s, ambiguous skips, missing items/domains)
