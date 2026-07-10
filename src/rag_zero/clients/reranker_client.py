"""OpenAI-compatible reranker using last-token P(Yes)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_zero.clients.base import BaseRerankerClient
from rag_zero.models.prompts import RERANKER_PROMPT

if TYPE_CHECKING:
    from rag_zero.clients.base import BaseLLMClient


class OpenAIRerankerClient(BaseRerankerClient):
    """Scores (query, document) pairs via a binary Yes/No prompt."""

    def __init__(self, llm_client: BaseLLMClient, max_tokens: int = 8) -> None:
        self.llm_client = llm_client
        self.max_tokens = max_tokens

    async def score(self, query: str, documents: list[str]) -> list[float]:
        """Return a score in [0, 1] for each document."""
        scores: list[float] = []
        for document in documents:
            prompt = RERANKER_PROMPT.format(query=query, document=document)
            response = (await self.llm_client.complete(prompt, max_tokens=self.max_tokens)).strip()
            text = response.lower()
            if text.startswith("yes"):
                scores.append(1.0)
            elif text.startswith("no"):
                scores.append(0.0)
            else:
                # Fallback: estimate from the presence of Yes/No tokens.
                scores.append(1.0 if "yes" in text else 0.0)
        return scores
