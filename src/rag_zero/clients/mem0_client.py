"""Optional mem0 memory service (Platform + OSS) with full CRUD.

Implements the latest mem0 Python SDK (mem0ai 2.x) API surface:
https://docs.mem0.ai/quickstart
https://docs.mem0.ai/open-source/python-quickstart

- Mem0 Platform: `from mem0 import MemoryClient` (requires API key).
- Mem0 OSS: `from mem0 import Memory` (self-hosted; defaults to SQLite).

All sync SDK calls are dispatched via ``asyncio.to_thread`` so this does not
block the async event loop. If ``mem0ai`` is not installed or fails to
initialize, the service degrades to a no-op with ``is_available()==False``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from rag_zero.config import Settings


class Mem0Service:
    """Full-featured mem0 client wrapper supporting both Platform and OSS."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any = None
        self._client_type: Literal["platform", "oss"] | None = None
        self._available = False
        self._initialize()

    def _initialize(self) -> None:
        try:
            from mem0 import Memory, MemoryClient
        except Exception:
            return

        try:
            if self.settings.mem0_api_key:
                kwargs: dict[str, Any] = {"api_key": self.settings.mem0_api_key}
                if self.settings.mem0_base_url:
                    kwargs["base_url"] = self.settings.mem0_base_url
                self._client = MemoryClient(**kwargs)
                self._client_type = "platform"
            else:
                self._client = Memory()
                self._client_type = "oss"
            self._available = True
        except Exception:
            self._client = None
            self._client_type = None
            self._available = False

    def is_available(self) -> bool:
        return self._available and self._client is not None

    @staticmethod
    def _normalize_search(result: Any) -> list[str]:
        """Pull memory text out of either Platform dicts or OSS lists."""
        if result is None:
            return []
        items = result.get("results", []) if isinstance(result, dict) else result
        if not isinstance(items, list):
            return []
        memories: list[str] = []
        for item in items:
            if isinstance(item, dict) and item.get("memory"):
                memories.append(str(item["memory"]))
            elif isinstance(item, str):
                memories.append(item)
        return memories

    @staticmethod
    def _normalize_list(result: Any) -> list[dict[str, Any]]:
        """Return ``get_all`` results as a list of memory dicts."""
        if result is None:
            return []
        items = result.get("results", []) if isinstance(result, dict) else result
        if not isinstance(items, list):
            return []
        return [i if isinstance(i, dict) else {"memory": i} for i in items]

    async def add(
        self,
        messages: list[dict[str, str]] | str,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> Any:
        """Store a conversation or fact for later retrieval.

        ``messages`` should normally be a list of ``{"role": "...", "content": "..."}``
        dicts; a raw string is accepted for the OSS SDK.
        """
        if not self.is_available():
            return None

        effective_user = user_id or self.settings.mem0_user_id

        def _add() -> Any:
            kwargs: dict[str, Any] = {}
            if effective_user:
                kwargs["user_id"] = effective_user
            if metadata:
                kwargs["metadata"] = metadata
            kwargs.update(extra)
            if self._client_type == "platform":
                return self._client.add(messages=messages, **kwargs)
            return self._client.add(messages, **kwargs)

        return await asyncio.to_thread(_add)

    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Return matching memory texts only (used by the agent pipeline)."""
        if not self.is_available():
            return []

        effective_user = user_id or self.settings.mem0_user_id
        top_k = limit if limit is not None else self.settings.memory_top_k

        def _search() -> list[str]:
            if self._client_type == "platform":
                filters: dict[str, Any] = {}
                if effective_user:
                    filters["user_id"] = effective_user
                result = self._client.search(query, filters=filters, top_k=top_k)
            else:
                kwargs: dict[str, Any] = {"limit": top_k}
                if effective_user:
                    kwargs["user_id"] = effective_user
                result = self._client.search(query, **kwargs)
            return self._normalize_search(result)

        return await asyncio.to_thread(_search)

    async def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List memories filtered by entity identifiers."""
        if not self.is_available():
            return []

        def _get_all() -> list[dict[str, Any]]:
            if self._client_type == "platform":
                filters: dict[str, Any] = {}
                for key, value in [
                    ("user_id", user_id or self.settings.mem0_user_id),
                    ("agent_id", agent_id),
                    ("app_id", app_id),
                    ("run_id", run_id),
                ]:
                    if value:
                        filters[key] = value
                result = self._client.get_all(filters=filters) if filters else self._client.get_all()
            else:
                kwargs: dict[str, Any] = {}
                effective_user = user_id or self.settings.mem0_user_id
                if effective_user:
                    kwargs["user_id"] = effective_user
                if agent_id:
                    kwargs["agent_id"] = agent_id
                if run_id:
                    kwargs["run_id"] = run_id
                result = self._client.get_all(**kwargs)
            return self._normalize_list(result)

        return await asyncio.to_thread(_get_all)

    async def update(
        self,
        memory_id: str,
        *,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Update the text and/or metadata of an existing memory."""
        if not self.is_available():
            return None
        if text is None and metadata is None:
            return None

        def _update() -> Any:
            kwargs: dict[str, Any] = {"memory_id": memory_id, "text": text}
            if metadata is not None:
                kwargs["metadata"] = metadata
            if self._client_type == "platform" and text is None:
                # Platform requires text; pass empty string if only metadata changed.
                kwargs["text"] = ""
            return self._client.update(**kwargs)

        return await asyncio.to_thread(_update)

    async def delete(self, memory_id: str) -> Any:
        """Delete a single memory by id."""
        if not self.is_available():
            return None

        def _delete() -> Any:
            if self._client_type == "platform":
                return self._client.delete(memory_id=memory_id)
            return self._client.delete(memory_id)

        return await asyncio.to_thread(_delete)

    async def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        run_id: str | None = None,
    ) -> Any:
        """Delete memories scoped to at least one entity identifier.

        Requires at least one filter (no global project wipe is allowed here).
        """
        if not self.is_available():
            return None
        if not any([user_id, agent_id, app_id, run_id]):
            return None

        def _delete_all() -> Any:
            if self._client_type == "platform":
                return self._client.delete_all(
                    user_id=user_id,
                    agent_id=agent_id,
                    app_id=app_id,
                    run_id=run_id,
                )
            kwargs: dict[str, Any] = {}
            if user_id:
                kwargs["user_id"] = user_id
            if agent_id:
                kwargs["agent_id"] = agent_id
            if app_id:
                kwargs["app_id"] = app_id
            if run_id:
                kwargs["run_id"] = run_id
            return self._client.delete_all(**kwargs)

        return await asyncio.to_thread(_delete_all)

    async def history(self, memory_id: str) -> list[dict[str, Any]]:
        """Fetch the change history of a memory, if supported by the SDK."""
        if not self.is_available():
            return []

        def _history() -> Any:
            if not hasattr(self._client, "history"):
                return []
            return self._client.history(memory_id=memory_id)

        result: Any = await asyncio.to_thread(_history)
        items = (
            cast("dict[str, Any]", result).get("history", [])
            if isinstance(result, dict)
            else result
        )
        if isinstance(items, list):
            return [i if isinstance(i, dict) else {"memory": i} for i in items]
        return []
