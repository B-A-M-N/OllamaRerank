# Security Notes

- Default bind is local-only. If you set host `0.0.0.0`, the server will refuse to start unless `RERANK_BIND_PUBLIC=1`. Do not expose it publicly without TLS/auth/rate limits (use your proxy/ingress).
- Use a firewall/allowlist for LAN exposure.
- No authentication is built in; wrap with your auth/identity layer if exposed beyond localhost.
- Avoid logging sensitive documents; tie-break logs can include prompt/raw snippets when debug flags are on.
