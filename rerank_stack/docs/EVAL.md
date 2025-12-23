# Quick Eval

Run the built-in labeled set to verify before/after metrics locally.

## Steps
1. Install deps and build the demo index (see README quickstart).
2. Start the API in another terminal (or rely on the fallback path for ambiguous queries).
3. Run the eval:
```bash
PYTHONPATH="$PWD/rerank_stack/src" python rerank_stack/src/rerank_demo/search.py --eval
```
This uses the bundled `EVAL_SET` (shower, car, bicycle, faucet, maintenance) and prints Accuracy@1, MRR, Recall@3 before/after rerank.

## Adding your own eval set
- Edit `EVAL_SET` in `rerank_stack/src/rerank_demo/search.py` with your query + relevant indices (matching `corpus.json` order).
- Re-run the command above to see metrics on your data.
