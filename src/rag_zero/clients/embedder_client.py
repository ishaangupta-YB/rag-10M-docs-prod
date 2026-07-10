"""OpenAI-compatible HTTP embedder client."""

from __future__ import annotations

import json

import httpx
import numpy as np
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag_zero.clients.base import BaseEmbedderClient


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def _is_transient_error(exc: BaseException) -> bool:
    """Return True for recoverable downstream failures."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 502, 503, 504}
    return isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.PoolTimeout,
        ),
    )


class OpenAIEmbedderClient(BaseEmbedderClient):
    """OpenAI-compatible `/embeddings` client for dense retrieval."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        expected_dim: int | None = None,
        query_instruction: str = "",
        batch_size: int = 32,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.expected_dim = expected_dim
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self._dim: int | None = None
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            http2=True,
            follow_redirects=False,
        )

    async def _call(self, texts: list[str]) -> list[list[float]]:
        async for attempt in AsyncRetrying(
            wait=wait_exponential_jitter(initial=1, max=10),
            stop=stop_after_attempt(self.max_retries),
            retry=retry_if_exception(_is_transient_error),
            reraise=True,
        ):
            with attempt:
                response = await self.client.post(
                    "/embeddings",
                    json={"model": self.model, "input": texts},
                )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code in {401, 403}:
                        raise RuntimeError(
                            "Authentication failed: check embedder API key"
                        ) from exc
                    raise
                try:
                    data = response.json()["data"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"Unexpected embedder response: {response.text}"
                    ) from exc

                if data and "index" in data[0]:
                    embeddings = sorted(data, key=lambda x: int(x["index"]))
                else:
                    embeddings = data
                return [item["embedding"] for item in embeddings]
        raise RuntimeError("Embedder request failed after retries")

    async def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if is_query and self.query_instruction:
            prefix = self.query_instruction
            if not prefix.endswith(" "):
                prefix += " "
            texts = [prefix + t for t in texts]

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_vectors.extend(await self._call(batch))

        matrix = np.array(all_vectors, dtype=np.float32)
        if matrix.size == 0:
            return matrix

        if self._dim is None and matrix.shape[1] > 0:
            self._dim = int(matrix.shape[1])
            if self.expected_dim is not None and self._dim != self.expected_dim:
                raise ValueError(
                    f"Embedder returned dimension {self._dim}, expected {self.expected_dim}"
                )

        return _normalize(matrix)

    @property
    def dim(self) -> int:
        if self._dim is None:
            from rag_zero.config import Settings

            return Settings().embedder_dim
        return self._dim

    async def close(self) -> None:
        await self.client.aclose()
