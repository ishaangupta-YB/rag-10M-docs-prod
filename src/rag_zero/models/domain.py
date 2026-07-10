"""Domain models for passages, chunks, answers, agent state, and evaluation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any

from typing_extensions import TypedDict


@dataclass(frozen=True, slots=True)
class Passage:
    """A raw passage from the source corpus."""

    id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk derived from a passage."""

    chunk_id: str
    passage_id: str
    title: str
    text: str
    context_prefix: str
    token_len: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def contextualized_text(self) -> str:
        return f"{self.context_prefix}\n{self.text}".strip()


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by the retrieval stack with scores."""

    chunk_id: str
    passage_id: str
    title: str
    text: str
    score: float
    retrieval_method: str


@dataclass(slots=True)
class CitedAnswer:
    """Intermediate answer with citations per sentence."""

    answer: str
    citations: list[str] = field(default_factory=list)
    method: str = ""
    latencies: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    """Final, user-facing answer status from the agent."""

    status: str  # answered, abstained
    answer: str
    citations: list[str]
    min_support: float
    reason: str
    latencies: dict[str, float]
    hops: int = 0

    def dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "answer": self.answer,
            "citations": self.citations,
            "min_support": self.min_support,
            "reason": self.reason,
            "latencies": self.latencies,
            "hops": self.hops,
        }


class AgentState(TypedDict):
    """LangGraph state for the CRAG agent.

    Lists use ``operator.add`` or the message reducer to accumulate across
    recursive retrieve-grade-refine hops.
    """

    question: str
    route: str
    query: str
    evidence: Annotated[list[RetrievedChunk], evidence_accumulator]
    grade: float
    draft: str
    gate: str
    final: FinalAnswer | None
    hops: int
    latencies: dict[str, float]
    memory_context: list[str]


def evidence_accumulator(
    existing: list[RetrievedChunk],
    new_chunks: list[RetrievedChunk] | RetrievedChunk,
) -> list[RetrievedChunk]:
    """Merge new evidence while deduplicating by chunk_id."""
    if isinstance(new_chunks, RetrievedChunk):
        new_chunks = [new_chunks]
    seen = {chunk.chunk_id for chunk in existing}
    merged = list(existing)
    for chunk in new_chunks:
        if chunk.chunk_id not in seen:
            merged.append(chunk)
            seen.add(chunk.chunk_id)
    return merged


@dataclass(frozen=True, slots=True)
class EvalItem:
    """A single evaluation example."""

    question: str
    gold_titles: list[str]
    gold_answer: str | None
    answerable: bool
    is_false_premise: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
