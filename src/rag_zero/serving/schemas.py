"""Pydantic request/response models for the FastAPI surface."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """POST /query request body."""

    question: str = Field(..., description="User question.")
    user_id: str | None = Field(default=None, description="Optional user/session id for mem0 memory recall.")
    use_memory: bool = Field(default=True, description="Enable memory recall/store if configured.")


class QueryResponse(BaseModel):
    """POST /query response body."""

    status: str = Field(..., description="answered or abstained")
    answer: str
    citations: list[str]
    min_support: float
    reason: str
    latencies: dict[str, float]
    hops: int


class IngestRequest(BaseModel):
    """POST /ingest request body."""

    source: str = Field(default="hotpotqa", description="Corpus source label.")
    slice: int | None = Field(default=None, description="Limit examples for hotpotqa.")


class IngestResponse(BaseModel):
    """POST /ingest response body."""

    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """GET /jobs/{id} response body."""

    job_id: str
    status: str
    progress: float = 0.0
    message: str


class EvaluateResponse(BaseModel):
    """POST /evaluate response body."""

    metrics: dict[str, float]
    confusion: dict[str, int]
    details: list[dict[str, Any]]


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str
    version: str


class MemoryMessage(BaseModel):
    """A single turn to persist to mem0."""

    role: str
    content: str


class MemoryAddRequest(BaseModel):
    """POST /memory/add request body."""

    messages: list[MemoryMessage] = Field(..., description="User/assistant turns to store.")
    user_id: str | None = Field(default=None)
    agent_id: str | None = Field(default=None)
    run_id: str | None = Field(default=None)
    metadata: dict[str, Any] | None = Field(default=None)


class MemoryAddResponse(BaseModel):
    """POST /memory/add response body."""

    stored: bool
    detail: str | None = None


class MemorySearchRequest(BaseModel):
    """POST /memory/search request body."""

    query: str
    user_id: str | None = Field(default=None)
    limit: int = Field(default=3, ge=1, le=50)


class MemorySearchResponse(BaseModel):
    """POST /memory/search response body."""

    memories: list[str]


class MemoryListResponse(BaseModel):
    """GET /memory response body."""

    memories: list[dict[str, Any]]


class MemoryUpdateRequest(BaseModel):
    """PATCH /memory/{memory_id} request body."""

    text: str | None = Field(default=None)
    metadata: dict[str, Any] | None = Field(default=None)


class MemoryUpdateResponse(BaseModel):
    """PATCH /memory/{memory_id} response body."""

    updated: bool
    detail: str | None = None


class MemoryDeleteResponse(BaseModel):
    """DELETE /memory/{memory_id} response body."""

    deleted: bool
    detail: str | None = None
