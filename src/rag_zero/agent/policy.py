"""Abstention policy producing typed FinalAnswer."""

from __future__ import annotations

from rag_zero.models.domain import CitedAnswer, FinalAnswer


class AbstentionPolicy:
    """Decide whether to answer or abstain based on grades, claims, and policy thresholds."""

    def __init__(self, tau_abstain: float = 0.5) -> None:
        self.tau_abstain = tau_abstain

    def decide(
        self,
        question: str,
        answer: CitedAnswer,
        status: str,
        min_support: float,
        hops: int,
        latencies: dict[str, float],
    ) -> FinalAnswer:
        if status == "verified":
            return FinalAnswer(
                status="answered",
                answer=answer.answer,
                citations=answer.citations,
                min_support=min_support,
                reason="verified",
                latencies=latencies,
                hops=hops,
            )
        if answer.answer.upper() == "INSUFFICIENT_EVIDENCE":
            return FinalAnswer(
                status="abstained",
                answer="I don't have enough evidence to answer that.",
                citations=[],
                min_support=0.0,
                reason="insufficient_evidence",
                latencies=latencies,
                hops=hops,
            )
        if status == "rejected":
            return FinalAnswer(
                status="abstained",
                answer="I could not verify the answer with the available evidence.",
                citations=answer.citations,
                min_support=min_support,
                reason="claim_verification_failed",
                latencies=latencies,
                hops=hops,
            )
        return FinalAnswer(
            status="abstained",
            answer="I don't have enough evidence to answer that.",
            citations=[],
            min_support=min_support,
            reason="abstention_policy",
            latencies=latencies,
            hops=hops,
        )
