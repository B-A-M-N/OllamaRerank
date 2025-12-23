# OllamaRerank

Precision-first reranking for embedding search. This stack takes your top‑K vector candidates, applies domain/item gates, scores what’s left with a lightweight reranker, and returns **FACTS** explaining every decision. The goal: fix “right domain, wrong object” mistakes (shower vs faucet, car vs bike) without adding heavy compute.

## Why this exists
Embedding retrieval is great at “nearby semantics,” but bad at **object identity**. Reranking fixes that:

- **Correct object wins** (shower cartridge beats kitchen faucet).
- **Wrong domain is rejected** (bicycle docs drop for car queries).
- **Explainable output** (every score has a reason).

## What it does
- **Hard gates**: item/domain mismatches become score `0` with `zero_reason`.
- **Coarse scoring**: small buckets (0/1/2/3) for usefulness.
- **Deterministic ordering**: accepted docs first; rejects tiered by reason, then similarity.
- **Ambiguity fallback**: vague/short queries skip the rerank API and fall back to similarity with lightweight heuristics.

## Quickstart
```bash
python -m venv venv
source venv/bin/activate
pip install -r rerank_stack/requirements.txt

# start API (adjust module/path if needed)
PYTHONPATH=rerank_stack/src uvicorn rerank_service.api:app --host 127.0.0.1 --port 8000 --reload

# run bundled demos
python rerank_stack/scripts/run_demo_cases.py
```

## Models
- Default demo model: `B-A-M-N/qwen3-reranker-0.6b-fp16` (Ollama).
- The server forwards the `model` string to Ollama `/api/generate`.
- You can swap in any reranker that returns the expected bucket contract.

## Policies
- Configs live in `rerank_stack/configs/*.yaml`.
- `policy_id` maps 1:1 to filenames (e.g., `howto_plumbing_v1` → `configs/howto_plumbing_v1.yaml`).
- Policies define domain anchors, item aliases, ambiguity triggers, and optional few‑shot examples.

## Custom data (quick example)
```bash
# build an embedding index from corpus.json
python rerank_stack/scripts/build_index.py \
  --model nomic-embed-text \
  --corpus-file corpus.json \
  --output-index index.npz \
  --output-docs docs.json

# run demos / eval
python rerank_stack/scripts/run_demo_cases.py
```

## Docs
Full guide lives at `rerank_stack/docs/USAGE.md` (pipeline, API schema, policy knobs, glossary, troubleshooting).

## License
MIT (see `LICENSE`).
