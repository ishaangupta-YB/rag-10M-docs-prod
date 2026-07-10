"""Unit tests for agent helpers."""

from __future__ import annotations

from rag_zero.agent.generator import CitedGenerator
from rag_zero.agent.policy import AbstentionPolicy
from rag_zero.agent.verifier import ClaimExtractor
from rag_zero.models.domain import CitedAnswer, FinalAnswer


def _make_generator() -> CitedGenerator:
    class _DummyLLM:
        async def chat(self, *args, **kwargs) -> str:
            return "Answer sentence one [a]. Answer sentence two [b]."

        async def complete(self, *args, **kwargs) -> str:
            return "Answer sentence one [a]. Answer sentence two [b]."

    return CitedGenerator(_DummyLLM())  # type: ignore[arg-type]


async def test_cited_generator_strips_invalid_citations() -> None:
    from rag_zero.models.domain import RetrievedChunk

    generator = _make_generator()
    evidence = [
        RetrievedChunk(
            chunk_id="a",
            passage_id="p1",
            title="T1",
            text="Evidence one.",
            score=1.0,
            retrieval_method="dense",
        ),
    ]
    result = await generator.generate("Q", evidence)
    assert "a" in result.citations
    assert "b" not in result.citations
    assert "[b]" not in result.answer


async def test_cited_generator_reports_insufficient_evidence() -> None:
    class _EmptyLLM:
        async def complete(self, *args, **kwargs) -> str:
            return "INSUFFICIENT_EVIDENCE"

    generator = CitedGenerator(_EmptyLLM())  # type: ignore[arg-type]
    result = await generator.generate("Q", [])
    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert result.citations == []


def test_claim_extractor_splits_sentences() -> None:
    claims = ClaimExtractor.extract("Claim one. Claim two. Claim three.")
    assert len(claims) == 3


def test_claim_extractor_returns_empty_on_abstain() -> None:
    assert ClaimExtractor.extract("INSUFFICIENT_EVIDENCE") == []


def test_abstention_policy_verified_returns_answered() -> None:
    policy = AbstentionPolicy(tau_abstain=0.5)
    answer = CitedAnswer(answer="The answer.", citations=["a"], method="generate")
    final = policy.decide(
        question="Q",
        answer=answer,
        status="verified",
        min_support=0.9,
        hops=0,
        latencies={},
    )
    assert isinstance(final, FinalAnswer)
    assert final.status == "answered"
    assert final.reason == "verified"


def test_abstention_policy_rejected_returns_abstained() -> None:
    policy = AbstentionPolicy(tau_abstain=0.5)
    answer = CitedAnswer(answer="The answer.", citations=["a"], method="generate")
    final = policy.decide(
        question="Q",
        answer=answer,
        status="rejected",
        min_support=0.2,
        hops=1,
        latencies={},
    )
    assert final.status == "abstained"
    assert final.reason == "claim_verification_failed"
