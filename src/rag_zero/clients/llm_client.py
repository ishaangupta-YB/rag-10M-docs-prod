"""OpenAI-compatible HTTP client with retries and timeouts."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag_zero.clients.base import BaseLLMClient, ChatMessage


def _is_transient_error(exc: BaseException) -> bool:
    """Return True for recoverable downstream failures that should be retried."""
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


class LLMClient(BaseLLMClient):
    """OpenAI-compatible chat completion client.

    Works against vLLM, OpenAI, Cloudflare AI Gateway, Ollama ``/v1``,
    or any other OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
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

    async def _request(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        async for attempt in AsyncRetrying(
            wait=wait_exponential_jitter(initial=1, max=10),
            stop=stop_after_attempt(self.max_retries),
            retry=retry_if_exception(_is_transient_error),
            reraise=True,
        ):
            with attempt:
                response = await self.client.post("/chat/completions", json=payload)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code in {401, 403}:
                        raise RuntimeError(
                            "Authentication failed: check RAG_LLM_API_KEY"
                        ) from exc
                    raise
                try:
                    data = response.json()
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Non-JSON response from LLM endpoint: {response.text}"
                    ) from exc
                return cast("dict[str, Any]", data)
        # Unreachable: reraise=True raises the last exception when retries are exhausted.
        raise RuntimeError("LLM request failed after retries")

    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.dict() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop

        data = await self._request(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected LLM response format: {data}") from exc
        if content is None:
            raise ValueError("LLM returned empty/refusal content")
        return str(content)

    async def json_prompt(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        content = await self.chat(messages, temperature=0.0, max_tokens=max_tokens)
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.lstrip("`").split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return cast("dict[str, Any]", json.loads(cleaned))
        except json.JSONDecodeError as exc:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return cast("dict[str, Any]", json.loads(cleaned[start : end + 1]))
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Could not parse JSON from response: {content}") from exc

    async def close(self) -> None:
        await self.client.aclose()
