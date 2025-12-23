import httpx
from typing import Dict, Any, Optional
import os
import traceback
import json

class OllamaClient:
    def __init__(self, base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")):
        self.base_url = base_url
        timeout = httpx.Timeout(connect=2.0, read=300.0, write=30.0, pool=2.0)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        self.client = httpx.AsyncClient(timeout=timeout, limits=limits)

    @staticmethod
    def _extract_text(data: Any) -> str:
        if isinstance(data, dict):
            if isinstance(data.get("response"), str):
                return data.get("response") or ""
            msg = data.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg.get("content") or ""
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0] or {}
                msg = first.get("message") if isinstance(first, dict) else {}
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg.get("content") or ""
                if isinstance(first.get("text"), str):
                    return first.get("text") or ""
            if isinstance(data.get("text"), str):
                return data.get("text") or ""
        if isinstance(data, str):
            # Sometimes the caller passes the raw text instead of parsed JSON.
            text_str = data.strip()
            if not text_str:
                return ""
            # Try to parse JSONL or a single JSON blob.
            for line in text_str.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    extracted = OllamaClient._extract_text(obj)
                    if extracted:
                        return extracted
                except Exception:
                    continue
            return text_str
        return ""

    async def generate(self, model: str, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options

        try:
            resp = await self.client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            text = self._extract_text(data)
            if not text and isinstance(resp.text, str) and resp.text.strip():
                # Fallback: maybe the text body has the content directly.
                text = resp.text

            if not text and os.getenv("RERANK_DEBUG_TIEBREAK", "").lower() in {"1", "true", "yes"}:
                print(
                    "[OLLAMA DEBUG EMPTY]",
                    {"status": resp.status_code, "headers": dict(resp.headers), "text_head": resp.text[:300], "json_keys": list(data.keys()) if isinstance(data, dict) else type(data)}
                )

            if text:
                return text.strip()

            return json.dumps(data)

        except httpx.RequestError as exc:
            print(f"Ollama HTTP Request Error: {exc}")
            traceback.print_exc()
            raise
        except httpx.HTTPStatusError as exc:
            print(f"Ollama HTTP Status Error {exc.response.status_code}: {exc}")
            print(f"Response text: {exc.response.text}")
            traceback.print_exc()
            raise
        except Exception as e:
            print(f"An unexpected error occurred during Ollama generation: {e}")
            traceback.print_exc()
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def __aenter__(self):
        return self
