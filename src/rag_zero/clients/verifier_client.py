"""LLM-as-judge verifier with optional local-model fallback."""

from __future__ import annotations

import asyncio
import re
from typing import cast

from rag_zero.clients.base import BaseLLMClient, BaseVerifierClient
from rag_zero.clients.local_hf_client import LocalVerifierClient
from rag_zero.models.prompts import JUDGE_PROMPT


class VerifierClient(BaseVerifierClient):
    """Verification client that uses an LLM judge and optional local NLI fallback."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        *,
        tau: float = 0.7,
        fallback: BaseVerifierClient | None = None,
        use_fallback: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.tau = tau
        self.fallback = fallback
        self.use_fallback = use_fallback

    @classmethod
    def from_local_nli(
        cls,
        llm_client: BaseLLMClient,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        tau: float = 0.7,
    ) -> VerifierClient:
        fallback: BaseVerifierClient = LocalVerifierClient(model_name=model_name)
        return cls(llm_client=llm_client, tau=tau, fallback=fallback)

    @staticmethod
    def _extract_score(text: str) -> float | None:
        """Extract the last float from the verifier response."""
        text = text.strip()
        if "CONTRADICTED" in text.upper() or "NO" in text.upper().split()[:2]:
            return 0.0
        numbers = [float(x) for x in re.findall(r"0?\.\d+|1\.0|0|1", text)]
        if numbers:
            return max(0.0, min(1.0, numbers[-1]))
        return None

    async def verify(self, claim: str, evidence: list[str]) -> float:
        if not evidence or not claim.strip():
            return 0.0

        evidence_text = "\n".join(f"- {e}" for e in evidence)
        prompt = JUDGE_PROMPT.format(claim=claim, evidence=evidence_text)
        try:
            response = await self.llm_client.complete(prompt, max_tokens=128)
        except Exception:
            response = ""

        score = self._extract_score(response)
        if score is None and self.use_fallback and self.fallback is not None:
            score = await self.fallback.verify(claim, evidence)
        return score if score is not None else 0.0

    async def verify_batch(self, claims: list[str], evidence: list[str]) -> list[float]:
        """Verify a batch of claims against the same evidence passage set."""
        if not evidence:
            return [0.0] * len(claims)
        evidence_text = "\n".join(evidence)
        tasks = [
            self.verify(claim, [evidence_text]) for claim in claims if claim.strip()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        output: list[float] = []
        for claim, result in zip(claims, results, strict=True):
            if isinstance(result, Exception) or claim.strip() == "":
                output.append(0.0)
            else:
                output.append(float(cast("float", result)))
        return output
