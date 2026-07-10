"""Reranking layer over fused candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_zero.models.domain import RetrievedChunk

if TYPE_CHECKING:
    from rag_zero.clients.base import BaseRerankerClient


class Reranker:
    """Rerank a candidate list using a cross-encoder."""

    def __init__(
        self,
        client: BaseRerankerClient,
        candidate_k: int = 150,
        top_n: int = 20,
    ) -> None:
        self.client = client
        self.candidate_k = candidate_k
        self.top_n = top_n

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        candidates = candidates[: self.candidate_k]
        if not candidates:
            return []

        scores = await self.client.score(query, [c.text for c in candidates])
        scored = list(zip(candidates, scores, strict=True))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                passage_id=c.passage_id,
                title=c.title,
                text=c.text,
                score=s,
                retrieval_method="rerank",
            )
            for c, s in scored[: self.top_n]
        ]
