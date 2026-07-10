"""Memory management endpoints backed by the optional mem0 service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request

from rag_zero.serving.schemas import (
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryDeleteResponse,
    MemoryListResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryUpdateRequest,
    MemoryUpdateResponse,
)

if TYPE_CHECKING:
    from rag_zero.clients.mem0_client import Mem0Service

router = APIRouter(prefix="/memory", tags=["memory"])


def get_memory_service(request: Request) -> Mem0Service:
    return cast("Mem0Service", request.app.state.memory_service)


def _ensure_available(service: Mem0Service) -> None:
    if not service.is_available():
        raise HTTPException(
            status_code=503,
            detail="Memory layer is not enabled or mem0ai is not configured",
        )


@router.post("/add", response_model=MemoryAddResponse)
async def add_memory(
    req: MemoryAddRequest,
    service: Mem0Service = Depends(get_memory_service),
) -> MemoryAddResponse:
    _ensure_available(service)
    messages: list[dict[str, str]] = [
        {"role": m.role, "content": m.content} for m in req.messages
    ]
    result = await service.add(
        messages=messages,
        user_id=req.user_id,
        metadata=req.metadata,
        agent_id=req.agent_id,
        run_id=req.run_id,
    )
    return MemoryAddResponse(stored=result is not None)


@router.post("/search", response_model=MemorySearchResponse)
async def search_memory(
    req: MemorySearchRequest,
    service: Mem0Service = Depends(get_memory_service),
) -> MemorySearchResponse:
    _ensure_available(service)
    memories = await service.search(
        req.query,
        user_id=req.user_id,
        limit=req.limit,
    )
    return MemorySearchResponse(memories=memories)


@router.get("", response_model=MemoryListResponse)
async def list_memories(
    user_id: str | None = None,
    agent_id: str | None = None,
    app_id: str | None = None,
    run_id: str | None = None,
    service: Mem0Service = Depends(get_memory_service),
) -> MemoryListResponse:
    _ensure_available(service)
    memories = await service.get_all(
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
    )
    return MemoryListResponse(memories=memories)


@router.patch("/{memory_id}", response_model=MemoryUpdateResponse)
async def update_memory(
    memory_id: str,
    req: MemoryUpdateRequest,
    service: Mem0Service = Depends(get_memory_service),
) -> MemoryUpdateResponse:
    _ensure_available(service)
    if req.text is None and req.metadata is None:
        raise HTTPException(
            status_code=422,
            detail="text or metadata must be provided",
        )
    result = await service.update(
        memory_id,
        text=req.text,
        metadata=req.metadata,
    )
    return MemoryUpdateResponse(updated=result is not None)


@router.delete("/{memory_id}", response_model=MemoryDeleteResponse)
async def delete_memory(
    memory_id: str,
    service: Mem0Service = Depends(get_memory_service),
) -> MemoryDeleteResponse:
    _ensure_available(service)
    result = await service.delete(memory_id)
    return MemoryDeleteResponse(deleted=result is not None)


@router.delete("/all")
async def delete_all_memories(
    user_id: str | None = None,
    agent_id: str | None = None,
    app_id: str | None = None,
    run_id: str | None = None,
    service: Mem0Service = Depends(get_memory_service),
) -> dict[str, Any]:
    _ensure_available(service)
    if not any([user_id, agent_id, app_id, run_id]):
        raise HTTPException(
            status_code=422,
            detail="at least one of user_id, agent_id, app_id, or run_id is required",
        )
    await service.delete_all(
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
    )
    return {"deleted": True}


@router.get("/{memory_id}/history")
async def memory_history(
    memory_id: str,
    service: Mem0Service = Depends(get_memory_service),
) -> dict[str, list[dict[str, Any]]]:
    _ensure_available(service)
    history = await service.history(memory_id)
    return {"history": history}
