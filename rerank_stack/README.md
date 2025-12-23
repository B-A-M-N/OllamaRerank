# OllamaRerank

Lightweight reranking stack for embedding-retrieved docs with hard domain/item gates, coarse buckets, and explainable FACTS.

## Quickstart

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src uvicorn rerank_service.api:app --host 127.0.0.1 --port 8000 --reload
python scripts/run_demo_cases.py
```

## Docs

See `docs/USAGE.md` for full pipeline walkthrough, configuration knobs, troubleshooting, and how to add your own data.

