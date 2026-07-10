"""Reciprocal Rank Fusion for dense + sparse rankings."""

from __future__ import annotations

from collections import defaultdict

from rag_zero.models.domain import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: dict[str, list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse multiple rankings into a single ordered list.

    Args:
        rankings: Mapping from method name to ordered list of chunks (best first).
        k: RRF constant.

    Returns:
        List of chunks sorted by RRF score descending, deduplicated by chunk_id.
    """
    scores: defaultdict[str, float] = defaultdict(float)
    chunks_by_id: dict[str, RetrievedChunk] = {}

    for _method, ranked in rankings.items():
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] += 1.0 / (k + rank)
            chunks_by_id[chunk.chunk_id] = chunk

    fused = sorted(
        chunks_by_id.values(),
        key=lambda c: (scores[c.chunk_id], c.score),
        reverse=True,
    )
    # Update score to be the RRF score.
    return [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            passage_id=c.passage_id,
            title=c.title,
            text=c.text,
            score=scores[c.chunk_id],
            retrieval_method="rrf",
        )
        for c in fused
    ]
