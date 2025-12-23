# Policy Setup Guide

This reranker is policy-driven. Each `policy_id` maps to a config in `rerank_stack/configs/*.yaml`. Follow these steps to define or tune a policy.

## Where to edit
- `rerank_stack/configs/*.yaml` (e.g., `howto_plumbing_v1.yaml`, `howto_automotive_v1.yaml`, `generic_v1.yaml`).
- `policy_id` in requests must match the filename (without extension).

## Required fields
- `object_groups` / `synonyms`: define items and aliases (e.g., shower, faucet, car, brake).
- `domain_keywords`: anchor words for the domain; used for domain match/gating.
- `topic_anchors` / `boost_anchors`: phrases that should boost relevance within the domain.
- `tier_a_keywords` / `tier_b_keywords`: used for tiering/gating.
- `intent_keywords`: verbs/nouns that signal procedural/diagnostic/overview intent.
- `caps`: hard gates (e.g., reject plumbing when query is automotive).
- `few_shot` (optional): query/doc/score examples to steer scoring.

## How scoring works
1) Gate by domain/item: set `zero_reason` for domain_mismatch/item_mismatch/no_item_found.
2) Bucket score (0/1/2/3) from policy logic.
3) Tie-break same-bucket items with a percent-based prompt (deterministic, temp=0, retry + fallback).
4) Stable sort fallback if tie-break is skipped/fails.

## Tuning for better behavior
- Broad queries (`maintenance`, `service`, `upkeep`): boost `kind=OVERVIEW`/`INFORMATIONAL` in intent fit to favor checklists/overviews over narrow procedures.
- Procedural queries: ensure `intent_keywords` include strong action verbs; keep `tier_a_keywords` tight.
- Missing item: default soft-demotes (`no_item_found`) instead of hard reject; adjust the demotion factor in `legacy_demo.py` if needed.

## Operational knobs (env)
- Concurrency/latency: `RERANK_TIE_MAX_CONCURRENCY`, `RERANK_TIE_TIMEOUT_SEC`, `RERANK_TIE_TOTAL_TIMEOUT_SEC`
- Cache: `RERANK_TIE_CACHE_TTL_SEC`
- Circuit breaker: `RERANK_TIE_CB_THRESHOLD`, `RERANK_TIE_CB_WINDOW_SEC`
- Doc clipping for tie prompt: `RERANK_TIE_DOC_CHARS`
- Metrics logging: `RERANK_TIE_METRICS=1` (prints cache hits, timeouts, retries, fallbacks)
- Skip ambiguous rerank: `RERANK_SKIP_AMBIGUOUS` (set `1` to use fallback similarity for ambiguous/short queries)

## Checklist for a new policy
1) Populate `object_groups`, `synonyms`, `domain_keywords`, `tier_a_keywords`, `tier_b_keywords`.
2) Add `caps` for obvious cross-domain rejections.
3) Add `few_shot` pairs for core query types.
4) Run `python rerank_stack/scripts/run_demo_cases.py --case <your_case>` with `PYTHONPATH=$PWD/rerank_stack/src` and inspect accept/reject reasons and tie values.
5) If broad queries look off, adjust intent fit/boosts for OVERVIEW/INFO as above.
