"""End-to-end hybrid retriever combining dense, sparse, RRF, and reranking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_zero.retrieval.dense import DenseRetriever
from rag_zero.retrieval.fusion import reciprocal_rank_fusion
from rag_zero.retrieval.rerank import Reranker
from rag_zero.retrieval.sparse import SparseRetriever

if TYPE_CHECKING:
    from rag_zero.clients.base import BaseEmbedderClient, BaseRerankerClient
    from rag_zero.config import Settings
    from rag_zero.ingestion.indexer import IndexStore
    from rag_zero.models.domain import RetrievedChunk


class HybridRetriever:
    """Dense + sparse + RRF + rerank retrieval pipeline."""

    def __init__(
        self,
        settings: Settings,
        store: IndexStore,
        embedder: BaseEmbedderClient,
        reranker_client: BaseRerankerClient,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.reranker = Reranker(
            reranker_client,
            candidate_k=150,
            top_n=settings.rerank_top_n,
        )
        self._dense: DenseRetriever | None = None
        self._sparse: SparseRetriever | None = None

    def _ensure_ready(self) -> None:
        try:
            _ = self.store.table
        except Exception as exc:
            raise RuntimeError(
                "Index not found. Run ingestion before querying."
            ) from exc

    @property
    def dense(self) -> DenseRetriever:
        if self._dense is None:
            self._ensure_ready()
            self._dense = DenseRetriever(self.store.table, self.embedder)
        return self._dense

    @property
    def sparse(self) -> SparseRetriever:
        if self._sparse is None:
            self._ensure_ready()
            self._sparse = SparseRetriever.from_index_store(self.store)
        return self._sparse

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        dense_results = await self.dense.search(query, k=self.settings.fusion_kk)
        sparse_results = await self.sparse.search(query, k=self.settings.fusion_kk)
        fused = reciprocal_rank_fusion(
            {"dense": dense_results, "sparse": sparse_results},
            k=self.settings.rrf_k,
        )
        return await self.reranker.rerank(query, fused)

    def recall_at_k(
        self,
        evidence: list[RetrievedChunk],
        gold_titles: list[str],
        k: int | None = None,
    ) -> float:
        """Fraction of unique gold titles present in top-k retrieved chunks."""
        if not gold_titles:
            return 1.0
        top_k = evidence[:k] if k is not None else evidence
        retrieved_titles = {c.title for c in top_k}
        found = sum(1 for title in gold_titles if title in retrieved_titles)
        return found / len(set(gold_titles))
