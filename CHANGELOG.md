# Changelog

## 0.1.0
- Initial public release with policy-driven rerank pipeline, tie-breaker fallback/retry, and production guardrails (timeouts, cache, circuit breaker, concurrency limits).
- Added docs for policy setup, usage, eval, integrations, and deployment modes.
- Dockerfile + docker-compose for local/LAN runs (bind-safe by default).
- CI workflow running pytest on PRs/pushes.
