"""MinHash-LSH near-duplicate removal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasketch import MinHash, MinHashLSH

if TYPE_CHECKING:
    from rag_zero.models.domain import Passage


def _shingles(text: str, k: int = 3) -> list[str]:
    """Return k-word shingles (preserves word boundaries)."""
    words = text.lower().split()
    if len(words) <= k:
        return [" ".join(words)]
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


def _make_minhash(text: str, num_perm: int, seed: int) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=seed)
    for s in _shingles(text):
        m.update(s.encode("utf-8"))
    return m


def deduplicate_passages(
    passages: list[Passage],
    threshold: float = 0.9,
    num_perm: int = 128,
    seed: int = 42,
) -> list[Passage]:
    """Remove near-duplicate passages using MinHash LSH.

    The first occurrence of each document ID is kept, then near-duplicate text
    is removed with deterministic MinHash permutations.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[str, MinHash] = {}
    kept: list[Passage] = []
    seen_ids: set[str] = set()

    for passage in passages:
        if passage.id in seen_ids:
            continue
        seen_ids.add(passage.id)
        mh = _make_minhash(passage.text, num_perm, seed)
        try:
            duplicates = lsh.query(mh)
        except Exception:
            # If LSH query fails, keep the passage rather than crashing.
            duplicates = []
        if duplicates:
            continue
        lsh.insert(passage.id, mh)
        minhashes[passage.id] = mh
        kept.append(passage)

    return kept
