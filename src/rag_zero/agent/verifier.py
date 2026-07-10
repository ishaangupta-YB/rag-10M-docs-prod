"""Claim extraction and verification gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_zero.clients.verifier_client import VerifierClient
    from rag_zero.models.domain import CitedAnswer, RetrievedChunk


class ClaimExtractor:
    """Split a free-text answer into atomic claims."""

    @staticmethod
    def extract(answer: str) -> list[str]:
        """Naive sentence-level claim extraction."""
        if not answer or answer.upper() == "INSUFFICIENT_EVIDENCE":
            return []
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        return sentences


class VerificationGate:
    """Verifies each claim against cited evidence; all must pass threshold."""

    def __init__(self, verifier: VerifierClient, tau: float = 0.7) -> None:
        self.verifier = verifier
        self.tau = tau

    async def check(
        self,
        answer: CitedAnswer,
        evidence: list[RetrievedChunk],
    ) -> tuple[str, float, float]:
        """Return (status, min_support, mean_support)."""
        if answer.answer.upper() == "INSUFFICIENT_EVIDENCE" or not answer.citations:
            return "abstain", 0.0, 0.0

        evidence_by_id = {c.chunk_id: c.text for c in evidence}
        claims = ClaimExtractor.extract(answer.answer)
        if not claims:
            return "abstain", 0.0, 0.0

        scores: list[float] = []
        for claim in claims:
            # Use only cited chunks as evidence for the claim.
            context = [evidence_by_id[cid] for cid in answer.citations if cid in evidence_by_id]
            score = await self.verifier.verify(claim, context)
            scores.append(score)

        min_support = min(scores)
        mean_support = sum(scores) / len(scores)
        if min_support >= self.tau:
            return "verified", min_support, mean_support
        return "rejected", min_support, mean_support
