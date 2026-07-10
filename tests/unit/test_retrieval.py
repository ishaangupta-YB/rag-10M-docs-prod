"""Unit tests for retrieval helpers."""

from __future__ import annotations

from rag_zero.models.domain import RetrievedChunk
from rag_zero.retrieval.fusion import reciprocal_rank_fusion
from rag_zero.retrieval.rerank import Reranker


class _DummyRerankerClient:
    async def score(self, query: str, documents: list[str]) -> list[float]:
        # Score by how often the query substring appears.
        return [float(doc.lower().count(query.lower())) for doc in documents]


def _chunk(chunk_id: str, text: str, title: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        passage_id=chunk_id.split("::")[0],
        title=title,
        text=text,
        score=0.0,
        retrieval_method="test",
    )


def test_reciprocal_rank_fusion_demotes_low_ranked() -> None:
    dense = [_chunk("a", "aaa"), _chunk("b", "bbb"), _chunk("c", "ccc")]
    sparse = [_chunk("b", "bbb"), _chunk("a", "aaa"), _chunk("d", "ddd")]
    fused = reciprocal_rank_fusion({"dense": dense, "sparse": sparse}, k=60)
    ids = [c.chunk_id for c in fused]
    # a and b appear in both rankings; d only once.
    assert "a" in ids[:3]
    assert "b" in ids[:3]
    assert "d" in ids


def test_reciprocal_rank_fusion_deduplicates() -> None:
    rankings = {
        "r1": [_chunk("x", "x"), _chunk("y", "y")],
        "r2": [_chunk("x", "x"), _chunk("y", "y")],
    }
    fused = reciprocal_rank_fusion(rankings, k=60)
    assert len(fused) == 2


async def test_reranker_truncates_candidates() -> None:
    client = _DummyRerankerClient()
    reranker = Reranker(client, candidate_k=3, top_n=2)
    candidates = [
        _chunk("1", "cat cat cat"),
        _chunk("2", "cat dog"),
        _chunk("3", "dog dog"),
        _chunk("4", "cat"),
    ]
    result = await reranker.rerank("cat", candidates)
    assert len(result) == 2
    assert result[0].chunk_id == "1"
    assert result[1].chunk_id == "2"


def test_reranker_empty_candidates() -> None:
    import asyncio

    async def _check() -> None:
        client = _DummyRerankerClient()
        reranker = Reranker(client, candidate_k=3, top_n=2)
        result = await reranker.rerank("cat", [])
        assert result == []

    asyncio.run(_check())
