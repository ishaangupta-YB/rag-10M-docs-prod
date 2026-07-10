"""Tests for the optional mem0 service wrapper (Platform + OSS)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag_zero.clients.mem0_client import Mem0Service
from rag_zero.config import Settings


def _settings(api_key: str | None = None) -> Settings:
    return Settings(mem0_api_key=api_key, mem0_user_id="default-user")


def _service(client_type: str, api_key: str | None = None) -> Mem0Service:
    service = object.__new__(Mem0Service)
    service.settings = _settings(api_key=api_key)
    service._client = MagicMock()
    service._client_type = client_type  # type: ignore[assignment]
    service._available = True
    return service


def test_service_unavailable_when_mem0ai_missing() -> None:
    service = Mem0Service(_settings())
    assert not service.is_available()


@pytest.mark.asyncio
async def test_search_platform_returns_memories() -> None:
    service = _service("platform", api_key="secret")
    service._client.search.return_value = {
        "results": [
            {"memory": "User likes Paris"},
            {"memory": "User prefers tea"},
            {"memory": None},
        ]
    }
    memories = await service.search("where does the user like", user_id="u1", limit=2)
    assert memories == ["User likes Paris", "User prefers tea"]
    service._client.search.assert_called_once_with(
        "where does the user like",
        filters={"user_id": "u1"},
        top_k=2,
    )


@pytest.mark.asyncio
async def test_search_oss_returns_memories() -> None:
    service = _service("oss")
    service._client.search.return_value = [
        {"memory": "Loves hiking"},
    ]
    memories = await service.search("hobbies", user_id="u1")
    assert memories == ["Loves hiking"]
    service._client.search.assert_called_once_with("hobbies", user_id="u1", limit=3)


@pytest.mark.asyncio
async def test_add_platform() -> None:
    service = _service("platform", api_key="secret")
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    await service.add(messages, user_id="u2", metadata={"foo": "bar"})
    service._client.add.assert_called_once_with(
        messages=messages,
        user_id="u2",
        metadata={"foo": "bar"},
    )


@pytest.mark.asyncio
async def test_add_oss() -> None:
    service = _service("oss")
    messages = [{"role": "user", "content": "Hi"}]
    await service.add(messages, user_id="u1")
    service._client.add.assert_called_once_with(messages, user_id="u1")


@pytest.mark.asyncio
async def test_get_all_platform() -> None:
    service = _service("platform", api_key="secret")
    service._client.get_all.return_value = {"results": [{"memory": "A"}]}
    results = await service.get_all(user_id="u1", agent_id="bot")
    assert results == [{"memory": "A"}]
    service._client.get_all.assert_called_once_with(
        filters={"user_id": "u1", "agent_id": "bot"},
    )


@pytest.mark.asyncio
async def test_get_all_oss() -> None:
    service = _service("oss")
    service._client.get_all.return_value = [{"memory": "B"}]
    results = await service.get_all(user_id="u1")
    assert results == [{"memory": "B"}]
    service._client.get_all.assert_called_once_with(user_id="u1")


@pytest.mark.asyncio
async def test_update_platform() -> None:
    service = _service("platform", api_key="secret")
    service._client.update.return_value = {"status": "ok"}
    result = await service.update("m1", text="new text", metadata={"k": "v"})
    assert result == {"status": "ok"}
    service._client.update.assert_called_once_with(
        memory_id="m1", text="new text", metadata={"k": "v"}
    )


@pytest.mark.asyncio
async def test_update_oss_metadata_only() -> None:
    service = _service("oss")
    await service.update("m1", metadata={"k": "v"})
    service._client.update.assert_called_once_with(
        memory_id="m1", text=None, metadata={"k": "v"}
    )


@pytest.mark.asyncio
async def test_update_noop_when_nothing_to_change() -> None:
    service = _service("platform")
    result = await service.update("m1")
    assert result is None
    service._client.update.assert_not_called()


@pytest.mark.asyncio
async def test_delete() -> None:
    service = _service("platform", api_key="secret")
    service._client.delete.return_value = {"deleted": True}
    result = await service.delete("m1")
    assert result == {"deleted": True}
    service._client.delete.assert_called_once_with(memory_id="m1")


@pytest.mark.asyncio
async def test_delete_all_requires_filter() -> None:
    service = _service("platform", api_key="secret")
    result = await service.delete_all()
    assert result is None
    service._client.delete_all.assert_not_called()


@pytest.mark.asyncio
async def test_delete_all_platform() -> None:
    service = _service("platform", api_key="secret")
    service._client.delete_all.return_value = {"deleted": 5}
    result = await service.delete_all(user_id="u1")
    service._client.delete_all.assert_called_once_with(
        user_id="u1", agent_id=None, app_id=None, run_id=None
    )
    assert result == {"deleted": 5}


@pytest.mark.asyncio
async def test_history_platform() -> None:
    service = _service("platform", api_key="secret")
    service._client.history.return_value = {"history": [{"event": "created"}]}
    history = await service.history("m1")
    assert history == [{"event": "created"}]


@pytest.mark.asyncio
async def test_history_unsupported() -> None:
    service = _service("oss")
    del service._client.history
    history = await service.history("m1")
    assert history == []
