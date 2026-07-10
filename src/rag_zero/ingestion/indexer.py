"""LanceDB dense + bm25s sparse index builder."""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING

import bm25s
import lancedb
import Stemmer

if TYPE_CHECKING:
    from rag_zero.clients.base import BaseEmbedderClient
    from rag_zero.config import Settings
    from rag_zero.models.domain import Chunk


class IndexStore:
    """Manages LanceDB and bm25s indices with build-or-load semantics."""

    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedderClient,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.uri = settings.normalized_lancedb_uri
        self.bm25_path = settings.normalized_bm25_path
        self.table_name = settings.lance_table
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.Table | None = None
        self._bm25: bm25s.BM25 | None = None
        self._corpus_ids: list[str] = []

    @property
    def db(self) -> lancedb.DBConnection:
        if self._db is None:
            self._db = lancedb.connect(str(self.uri))
        return self._db

    def exists(self) -> bool:
        if not self.uri.exists() or not (self.bm25_path / "params.json").exists():
            return False
        try:
            table = self.db.open_table(self.table_name)
            rows_ok = table.count_rows() > 0
            bm25_ok = (self.bm25_path / "params.json").exists()
        except Exception:
            return False
        return rows_ok and bm25_ok

    async def build(self, chunks: list[Chunk], overwrite: bool = False) -> IndexStore:
        """Build both indices from chunks."""
        if not chunks:
            raise ValueError("Cannot build an index from an empty chunk list")

        texts = [c.contextualized_text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        passages_ids = [c.passage_id for c in chunks]
        titles = [c.title for c in chunks]

        embeddings = await self.embedder.encode(texts, is_query=False)
        expected_dim = self.settings.embedder_dim
        actual_dim = embeddings.shape[1]
        if actual_dim != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: got {actual_dim}, expected {expected_dim}"
            )

        # Only drop old artifacts after the expensive embedding step succeeds.
        if overwrite:
            self._drop_existing()

        records = [
            {
                "chunk_id": ids[i],
                "passage_id": passages_ids[i],
                "title": titles[i],
                "text": texts[i],
                "vector": embeddings[i].tolist(),
            }
            for i in range(len(chunks))
        ]

        table = self.db.create_table(self.table_name, data=records[:1], mode="overwrite")
        batch_size = max(1024, min(50_000, len(records) // 10 + 1))
        for i in range(0, len(records), batch_size):
            table.add(records[i : i + batch_size])

        if len(chunks) >= 256:
            self._create_vector_index(table, len(chunks))
        self._table = table

        stemmer = Stemmer.Stemmer("english")
        tokenized = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)
        bm25_obj = bm25s.BM25()
        bm25_obj.index(tokenized)
        self.bm25_path.mkdir(parents=True, exist_ok=True)
        bm25_obj.save(str(self.bm25_path), corpus=texts)
        self.bm25_path.mkdir(parents=True, exist_ok=True)
        metadata = [
            {
                "id": ids[i],
                "passage_id": passages_ids[i],
                "title": titles[i],
                "text": texts[i],
            }
            for i in range(len(ids))
        ]
        (self.bm25_path / "metadata.jsonl").write_text(
            "\n".join(json.dumps(rec) for rec in metadata),
            encoding="utf-8",
        )
        (self.bm25_path / "corpus.jsonl").write_text(
            "\n".join(json.dumps({"id": rec["id"], "text": rec["text"]}) for rec in metadata),
            encoding="utf-8",
        )

        self._bm25 = bm25_obj
        self._corpus_ids = ids
        return self

    def _create_vector_index(self, table: lancedb.Table, num_rows: int) -> None:
        """Build an IVF_PQ index tuned for the corpus size."""
        # Rule of thumb: 4 * sqrt(N) partitions, bounded by [32, 4096].
        num_partitions = max(32, min(4096, int(4 * (num_rows**0.5))))
        # Sub-vectors should divide the embedding dimension evenly.
        dim = self.settings.embedder_dim
        candidates = [s for s in (8, 16, 32, 64, 128) if dim % s == 0]
        num_sub_vectors = candidates[-1] if candidates else dim
        try:
            table.create_index(
                vector_column_name="vector",
                index_type="IVF_PQ",
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
            )
        except TypeError:
            # Newer LanceDB uses a config object; fall back to defaults.
            table.create_index("vector", index_type="IVF_PQ")

    def _drop_existing(self) -> None:
        with contextlib.suppress(Exception):
            self.db.drop_table(self.table_name)
        if (self.bm25_path / "params.json").exists():
            import shutil

            shutil.rmtree(self.bm25_path, ignore_errors=True)

    def load(self) -> IndexStore:
        """Load existing indices from disk."""
        self._table = self.db.open_table(self.table_name)
        self._bm25 = bm25s.BM25.load(str(self.bm25_path), load_corpus=True)
        corpus_file = self.bm25_path / "corpus.jsonl"
        self._corpus_ids = []
        if corpus_file.exists():
            with corpus_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        self._corpus_ids.append(str(obj["id"]))
        return self

    @property
    def table(self) -> lancedb.Table:
        if self._table is None:
            self.load()
        return self._table

    @property
    def bm25(self) -> bm25s.BM25:
        if self._bm25 is None:
            self.load()
        return self._bm25

    @property
    def corpus_ids(self) -> list[str]:
        if not self._corpus_ids:
            self.load()
        return self._corpus_ids

    async def build_or_load(
        self, chunks: list[Chunk], overwrite: bool = False
    ) -> IndexStore:
        if not overwrite and self.exists():
            return self.load()
        return await self.build(chunks, overwrite=overwrite)
