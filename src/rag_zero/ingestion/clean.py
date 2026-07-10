"""Text normalization and quality filtering."""

from __future__ import annotations

import re
import unicodedata

from rag_zero.models.domain import Passage


def normalize_text(text: str) -> str:
    """Apply NFKC normalization, remove soft hyphens, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_quality(text: str, min_words: int = 20) -> bool:
    """Basic quality gate: require a minimum number of words."""
    return len(text.split()) >= min_words


def clean_passages(passages: list[Passage], min_words: int = 20) -> list[Passage]:
    """Normalize and filter a list of passages."""
    cleaned: list[Passage] = []
    for passage in passages:
        norm = normalize_text(passage.text)
        if not is_quality(norm, min_words=min_words):
            continue
        cleaned.append(
            Passage(
                id=passage.id,
                title=normalize_text(passage.title),
                text=norm,
                metadata=passage.metadata,
            )
        )
    return cleaned
