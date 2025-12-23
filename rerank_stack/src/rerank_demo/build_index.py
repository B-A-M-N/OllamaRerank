import httpx
import json
import numpy as np
import asyncio
from typing import List, Dict, Any
from pathlib import Path # Import Path

class OllamaEmbeddingClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/api/embeddings"
        embeddings_list = []
        for text in texts:
            payload = {
                "model": model,
                "prompt": text,
            }
            try:
                response = await self.client.post(url, json=payload, timeout=60.0)
                response.raise_for_status()
                result = response.json()
                embeddings_list.append(result.get("embedding", []))
            except httpx.RequestError as exc:
                print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
                raise
            except httpx.HTTPStatusError as exc:
                print(
                    f"Error response {exc.response.status_code} while requesting {exc.request.url!r}: {exc.response.text}"
                )
                raise
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise
        return embeddings_list

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def __aenter__(self):
        return self


async def build_index(
    corpus_path: Path = Path(__file__).resolve().parent.parent.parent / "corpus.json",
    output_vectors_path: Path = Path(__file__).resolve().parent.parent.parent / "index.npz",
    output_docs_path: Path = Path(__file__).resolve().parent.parent.parent / "docs.json",
    embedding_model: str = "nomic-embed-text",
):
    print(f"Building embedding index using model: {embedding_model}")
    print(f"Loading corpus from {corpus_path}")
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    if not corpus:
        print("Corpus is empty. Exiting.")
        return

    print(f"Loaded {len(corpus)} documents. Generating embeddings...")

    client = OllamaEmbeddingClient()
    try:
        embeddings = []
        for i, doc in enumerate(corpus):
            print(f"  Embedding document {i+1}/{len(corpus)}...")
            try:
                embedding = await client.embed(embedding_model, [doc])
                embeddings.append(embedding[0]) # embed returns List[List[float]], we need List[float]
            except Exception as e:
                print(f"Failed to embed document {i}: {e}. Skipping.")
                embeddings.append(np.zeros(768).tolist()) # Append a zero vector as a placeholder
    finally:
        await client.aclose() # Ensure client is closed

    if not embeddings:
        print("No embeddings generated. Exiting.")
        return

    embeddings_array = np.array(embeddings, dtype=np.float32)
    print(f"Embeddings generated. Shape: {embeddings_array.shape}")

    np.savez_compressed(output_vectors_path, embeddings=embeddings_array)
    print(f"Embeddings saved to {output_vectors_path}")

    # Save documents alongside their original index
    with open(output_docs_path, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Documents saved to {output_docs_path}")
    print("Index build complete.")

if __name__ == "__main__":
    # Ensure an Ollama server is running and the embedding model is pulled:
    # ollama run nomic-embed-text
    asyncio.run(build_index())