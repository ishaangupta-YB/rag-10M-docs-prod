"""Unit tests for ingestion helpers."""

from __future__ import annotations

from rag_zero.ingestion.chunker import SentenceChunker
from rag_zero.ingestion.clean import clean_passages, normalize_text
from rag_zero.ingestion.dedup import deduplicate_passages
from rag_zero.models.domain import Passage


class _MockTokenizer:
    """Simple whitespace tokenizer for deterministic chunking tests."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, tokens: list[int], *, skip_special_tokens: bool = True) -> str:
        return " ".join(str(t) for t in tokens)


class _PassthroughTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return text.split()

    def decode(self, tokens: list[str], *, skip_special_tokens: bool = True) -> str:
        return " ".join(str(t) for t in tokens)


class _SentenceChunkerForTest(SentenceChunker):
    def __init__(self, target_tokens: int = 10, overlap_tokens: int = 2) -> None:
        self.tokenizer_name = "mock"
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = _MockTokenizer()


class _PassthroughChunkerForTest(SentenceChunker):
    def __init__(self, target_tokens: int = 5, overlap_tokens: int = 1) -> None:
        self.tokenizer_name = "passthrough"
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = _PassthroughTokenizer()


def test_normalize_text_handles_unicode_and_whitespace() -> None:
    raw = "caf\u00e9\u00ad\n\n   bar"
    assert normalize_text(raw) == "café bar"


def test_clean_passages_filters_short_texts() -> None:
    passages = [
        Passage(id="1", title="A", text="Short text.", metadata={}),
        Passage(id="2", title="B", text=" " * 5, metadata={}),
        Passage(id="3", title="C", text="This is a long enough passage with many words to pass the filter.", metadata={}),
    ]
    cleaned = clean_passages(passages, min_words=5)
    assert len(cleaned) == 1
    assert cleaned[0].id == "3"


def test_deduplicate_passages_removes_exact_duplicates() -> None:
    base_text = "The quick brown fox jumps over the lazy dog. " * 10
    passages = [
        Passage(id="a", title="A", text=base_text, metadata={}),
        Passage(id="b", title="B", text=base_text, metadata={}),
        Passage(id="c", title="C", text="Something completely different and unrelated to the others.", metadata={}),
    ]
    kept = deduplicate_passages(passages, threshold=0.9)
    assert len(kept) == 2
    kept_ids = {p.id for p in kept}
    assert "a" in kept_ids
    assert "c" in kept_ids


def test_chunker_respects_token_budget() -> None:
    passage = Passage(
        id="doc",
        title="T",
        text="Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six.",
        metadata={},
    )
    chunker = _PassthroughChunkerForTest(target_tokens=5, overlap_tokens=1)
    chunks = chunker.chunk_passage(passage)
    assert len(chunks) >= 2
    for chunk in chunks:
        # The passthrough tokenizer splits on whitespace, so token_len equals word count.
        # Overlap may push a chunk slightly above the target; allow a small margin.
        assert chunk.token_len <= chunker.target_tokens + chunker.overlap_tokens + 2


def test_chunk_ids_are_unique_and_reference_passage() -> None:
    passage = Passage(id="p1", title="T", text="A. B. C. D. E. F. G. H. I. J.", metadata={})
    chunker = _SentenceChunkerForTest(target_tokens=3, overlap_tokens=1)
    chunks = chunker.chunk_passage(passage)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    for chunk in chunks:
        assert chunk.passage_id == "p1"
        assert chunk.context_prefix == ""
