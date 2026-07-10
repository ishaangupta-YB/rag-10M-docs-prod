"""Abstract model client interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np  # noqa: TC002


class ChatMessage:
    """A single chat message."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class BaseLLMClient(ABC):
    """Abstract interface for any chat-completion LLM client."""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Return raw text completion."""

    @abstractmethod
    async def json_prompt(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-parsed object."""

    async def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Convenience wrapper for a single user prompt."""
        return await self.chat(
            [ChatMessage(role="user", content=prompt)],
            max_tokens=max_tokens,
        )


class BaseEmbedderClient(ABC):
    """Abstract embedder client."""

    @abstractmethod
    async def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        """Return normalized embeddings of shape (len(texts), dim)."""

    async def encode_query(self, query: str) -> np.ndarray:
        encoded = await self.encode([query], is_query=True)
        return cast("np.ndarray", encoded[0])


class BaseRerankerClient(ABC):
    """Abstract reranker client returning scores in [0, 1]."""

    @abstractmethod
    async def score(self, query: str, documents: list[str]) -> list[float]:
        """Return relevance score for each document."""


class BaseVerifierClient(ABC):
    """Abstract claim verifier client."""

    @abstractmethod
    async def verify(self, claim: str, evidence: list[str]) -> float:
        """Return score in [0, 1] indicating claim support."""
