import asyncio
from pathlib import Path
import sys

# Ensure local src is on path for direct imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root / "rerank_stack" / "src"))

from rerank_service.scoring import fine_score_documents
from rerank_service.parser import parse_fine_score


class _FakeOllamaClient:
    def __init__(self, response: str):
        self.response = response

    async def generate(self, model: str, prompt: str, options: dict) -> str:
        # Mimic async API: ignore inputs, return fixed response
        return self.response


def test_fine_score_documents_parses_single_line_score():
    client = _FakeOllamaClient("FINE_SCORE: 77")
    docs = [(123, "Example document text.")]

    results = asyncio.run(
        fine_score_documents(
            ollama_client=client,
            model="dummy-model",
            query="dummy query",
            docs=docs,
        )
    )

    assert 123 in results
    meta = results[123]
    assert meta["fine_score"] == 77
    assert abs(meta["fine_score_norm"] - 0.77) < 1e-6
    assert meta["error"] is None


def test_fine_score_documents_handles_empty_output():
    client = _FakeOllamaClient("")
    docs = [(1, "Doc")]

    results = asyncio.run(
        fine_score_documents(
            ollama_client=client,
            model="dummy-model",
            query="q",
            docs=docs,
        )
    )

    assert results[1]["fine_score"] is None
    assert results[1]["error"] == "empty_output"


def test_parse_fine_score_tolerates_percent_and_float():
    cases = {
        "FINE_SCORE: 70%": 70,
        "score=0.72": 72,
        "42": 42,
        "0.3": 30,
        "fine score: 99": 99,
        "TIE: 70%": 70,
        "TIE: 0.65": 65,
        '{"tie": 0.5}': 50,
    }
    for raw, expected in cases.items():
        val, err = parse_fine_score(raw)
        assert err is None
        assert val == expected
