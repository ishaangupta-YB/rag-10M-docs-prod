"""Sentence-aware chunking with a tokenizer token budget."""

from __future__ import annotations

import nltk
from transformers import AutoTokenizer

from rag_zero.models.domain import Chunk, Passage


def _download_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def _split_sentences(text: str) -> list[str]:
    _download_punkt()
    try:
        return [str(s) for s in nltk.sent_tokenize(text)]
    except Exception:
        # Fallback for environments without NLTK data.
        import re

        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class SentenceChunker:
    """Chunk passages into token-budgeted pieces respecting sentence boundaries."""

    def __init__(
        self,
        tokenizer_name: str,
        target_tokens: int = 256,
        overlap_tokens: int = 32,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_passage(self, passage: Passage) -> list[Chunk]:
        sentences = _split_sentences(passage.text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sent_len = self._token_count(sentence)
            if current_len + sent_len > self.target_tokens and current:
                chunks.append(self._build_chunk(passage, current, len(chunks)))
                overlap_text = self._sentence_overlap(current)
                current = [overlap_text] if overlap_text else []
                current_len = self._token_count(" ".join(current)) if current else 0

            current.append(sentence)
            current_len += sent_len

        if current and any(s.strip() for s in current):
            chunks.append(self._build_chunk(passage, current, len(chunks)))

        return chunks

    def _build_chunk(self, passage: Passage, sentences: list[str], index: int) -> Chunk:
        text = " ".join(sentences).strip()
        token_len = self._token_count(text)
        return Chunk(
            chunk_id=f"{passage.id}::{index}",
            passage_id=passage.id,
            title=passage.title,
            text=text,
            context_prefix="",
            token_len=token_len,
            metadata={"source": passage.metadata.get("source", "")},
        )

    def _sentence_overlap(self, sentences: list[str]) -> str:
        """Return whole-sentence overlap whose token length <= overlap_tokens."""
        overlap: list[str] = []
        overlap_len = 0
        for sentence in reversed(sentences):
            sent_len = self._token_count(sentence)
            if overlap_len + sent_len > self.overlap_tokens:
                break
            overlap.insert(0, sentence)
            overlap_len += sent_len
        if not overlap and sentences:
            # If the last single sentence exceeds the overlap budget, include it anyway
            # so that consecutive chunks share at least one sentence.
            overlap = [sentences[-1]]
        return " ".join(overlap)

    def chunk_passages(self, passages: list[Passage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for passage in passages:
            chunks.extend(self.chunk_passage(passage))
        return chunks


def chunk_passages(
    passages: list[Passage],
    tokenizer_name: str,
    target_tokens: int = 256,
    overlap_tokens: int = 32,
) -> list[Chunk]:
    """Top-level convenience function."""
    chunker = SentenceChunker(tokenizer_name, target_tokens, overlap_tokens)
    return chunker.chunk_passages(passages)
