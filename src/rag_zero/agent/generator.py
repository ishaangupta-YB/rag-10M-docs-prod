"""CitedGenerator: context-only answer generation with citation stripping."""

from __future__ import annotations

import re

from rag_zero.clients.base import BaseLLMClient, ChatMessage
from rag_zero.models.domain import CitedAnswer, RetrievedChunk
from rag_zero.models.prompts import GENERATOR_PROMPT


class CitedGenerator:
    """Generates answers with one citation per sentence and strips invented citations."""

    def __init__(self, llm_client: BaseLLMClient, max_tokens: int = 1024) -> None:
        self.llm_client = llm_client
        self.max_tokens = max_tokens

    async def generate(
        self,
        question: str,
        evidence: list[RetrievedChunk],
    ) -> CitedAnswer:
        if not evidence:
            return CitedAnswer(
                answer="INSUFFICIENT_EVIDENCE",
                citations=[],
                method="generate",
            )

        evidence_text = self._format_evidence(evidence)
        prompt = GENERATOR_PROMPT.format(question=question, evidence=evidence_text)
        response = await self.llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        answer = response.strip()
        valid_ids = {c.chunk_id for c in evidence}
        raw_citations = self._extract_citations(answer)
        citations = [cid for cid in raw_citations if cid in valid_ids]
        cleaned = self._strip_invalid_citations(answer, valid_ids)

        if "INSUFFICIENT_EVIDENCE" in cleaned.upper() or not citations:
            return CitedAnswer(
                answer="INSUFFICIENT_EVIDENCE",
                citations=[],
                method="generate",
            )

        return CitedAnswer(answer=cleaned, citations=citations, method="generate")

    def _format_evidence(self, evidence: list[RetrievedChunk]) -> str:
        lines: list[str] = []
        for chunk in evidence:
            lines.append(f"[{chunk.chunk_id}] {chunk.title}\n{chunk.text}")
        return "\n\n".join(lines)

    def _extract_citations(self, text: str) -> list[str]:
        return list(re.findall(r"\[([A-Za-z0-9_:\-]+)\]", text))

    def _strip_invalid_citations(self, text: str, valid_ids: set[str]) -> str:
        """Remove any sentence whose only citation is to an invalid chunk_id."""
        # Split on sentence boundaries while keeping delimiters.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept: list[str] = []
        for sentence in sentences:
            ids = set(self._extract_citations(sentence))
            if ids and ids.isdisjoint(valid_ids):
                continue
            kept.append(sentence)
        return " ".join(kept).strip()
