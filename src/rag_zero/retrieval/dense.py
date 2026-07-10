"""Dense vector search over a LanceDB table."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_zero.models.domain import RetrievedChunk

if TYPE_CHECKING:
    import lancedb

    from rag_zero.clients.base import BaseEmbedderClient


def _cosine_distance_to_similarity(distance: float) -> float:
    """LanceDB cosine distance is in [0, 2]; map to similarity in [0, 1]."""
    return max(0.0, 1.0 - distance / 2.0)


class DenseRetriever:
    """Cosine-similarity dense retriever backed by LanceDB."""

    def __init__(
        self,
        table: lancedb.Table,
        embedder: BaseEmbedderClient,
    ) -> None:
        self.table = table
        self.embedder = embedder

    # Default IVF-PQ probe count. For 10M+ docs, raise this via settings or caller.
    DEFAULT_NPROBES = 64

    async def search(
        self,
        query: str,
        k: int = 20,
        refine_factor: int = 10,
        nprobes: int | None = None,
    ) -> list[RetrievedChunk]:
        vector = await self.embedder.encode_query(query)
        vector_list = vector.tolist()
        probes = nprobes if nprobes is not None else self.DEFAULT_NPROBES
        results = (
            self.table.search(vector_list)
            .metric("cosine")
            .limit(k)
            .nprobes(probes)
            .refine_factor(refine_factor)
            .to_pandas()
        )

        chunks: list[RetrievedChunk] = []
        for _, row in results.iterrows():
            distance = float(row.get("_distance", row.get("distance", 0.0)))
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(row["chunk_id"]),
                    passage_id=str(row["passage_id"]),
                    title=str(row["title"]),
                    text=str(row["text"]),
                    score=_cosine_distance_to_similarity(distance),
                    retrieval_method="dense",
                )
            )
        return chunks
