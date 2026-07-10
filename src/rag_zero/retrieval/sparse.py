"""Sparse lexical search with bm25s."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import bm25s
import Stemmer

from rag_zero.models.domain import RetrievedChunk

if TYPE_CHECKING:
    from pathlib import Path

    from rag_zero.ingestion.indexer import IndexStore


class SparseRetriever:
    """BM25 sparse retriever backed by bm25s."""

    def __init__(
        self,
        bm25: bm25s.BM25,
        corpus_texts: list[str],
        corpus_ids: list[str],
        corpus_passage_ids: list[str],
        corpus_titles: list[str],
    ) -> None:
        self.bm25 = bm25
        self.corpus_texts = corpus_texts
        self.corpus_ids = corpus_ids
        self.corpus_passage_ids = corpus_passage_ids
        self.corpus_titles = corpus_titles
        self.stemmer = Stemmer.Stemmer("english")
        self._id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    @classmethod
    def from_index_store(cls, store: IndexStore) -> SparseRetriever:
        """Build a sparse retriever from an IndexStore using persisted metadata."""
        from rag_zero.ingestion.indexer import IndexStore

        assert isinstance(store, IndexStore)
        ids = store.corpus_ids
        meta_path = store.bm25_path / "metadata.jsonl"
        if meta_path.exists():
            rows = _load_metadata(meta_path)
        else:
            # Fallback for legacy artifacts; avoid on large corpora.
            rows = {
                str(row["chunk_id"]): row
                for _, row in store.table.to_pandas().iterrows()
            }
        texts = [str(rows[cid]["text"]) for cid in ids if cid in rows]
        passage_ids = [str(rows[cid]["passage_id"]) for cid in ids if cid in rows]
        titles = [str(rows[cid]["title"]) for cid in ids if cid in rows]
        return cls(store.bm25, texts, ids, passage_ids, titles)

    async def search(self, query: str, k: int = 20) -> list[RetrievedChunk]:
        tokenized = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer)
        results, scores = self.bm25.retrieve(
            tokenized,
            k=min(k, len(self.corpus_texts)),
            k_tokens=False,
        )
        chunks: list[RetrievedChunk] = []
        for i in range(results.shape[1]):
            idx = int(results[0, i])
            score = float(scores[0, i])
            chunks.append(
                RetrievedChunk(
                    chunk_id=self.corpus_ids[idx],
                    passage_id=self.corpus_passage_ids[idx],
                    title=self.corpus_titles[idx],
                    text=self.corpus_texts[idx],
                    score=score,
                    retrieval_method="sparse",
                )
            )
        return chunks


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[str(obj["id"])] = {
                "text": str(obj.get("text", "")),
                "passage_id": str(obj.get("passage_id", "")),
                "title": str(obj.get("title", "")),
            }
    return rows
