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

            # /api/generate
            if "response" in data:
                return (data.get("response") or "").strip()

            # Fallback for /api/chat-like responses, just in case
            msg = data.get("message") or {}
            if isinstance(msg, dict) and "content" in msg:
                return (msg.get("content") or "").strip()

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