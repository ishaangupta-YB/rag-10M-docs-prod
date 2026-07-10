"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator  # noqa: TC003
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag_zero.config import Settings
from rag_zero.infra.logging import configure_logging
from rag_zero.infra.tracing import configure_tracing
from rag_zero.serving.memory_routes import router as memory_router
from rag_zero.serving.routes import router
from rag_zero.serving.service import EvaluationService, IngestionService, QueryService


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings: Settings = app.state.settings
    settings.ensure_dirs()
    configure_logging(settings.log_level)
    configure_tracing(settings)
    app.state.query_service = QueryService(settings)
    app.state.ingestion_service = IngestionService(settings)
    app.state.evaluation_service = EvaluationService(
        settings, query_service=app.state.query_service
    )
    app.state.memory_service = app.state.query_service.memory_service
    yield
    await app.state.query_service.close()


def _setup_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RuntimeError)
    async def _runtime_error_handler(_request: Request, exc: RuntimeError) -> JSONResponse:
        message = str(exc)
        status_code = 503 if "index" in message.lower() else 500
        return JSONResponse(
            status_code=status_code,
            content={"detail": message, "code": "runtime_error"},
        )

    @app.exception_handler(httpx.TimeoutException)
    async def _timeout_handler(_request: Request, _exc: httpx.TimeoutException) -> JSONResponse:
        return JSONResponse(
            status_code=504,
            content={"detail": "downstream model timeout", "code": "model_timeout"},
        )

    @app.exception_handler(httpx.ConnectError)
    async def _connect_handler(_request: Request, _exc: httpx.ConnectError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"detail": "cannot connect to model endpoint", "code": "model_connect_error"},
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()
    app = FastAPI(
        title="RAG Zero",
        version="0.1.0",
        description="Production-grade near-zero-hallucination agentic RAG",
        lifespan=_lifespan,
    )
    app.state.settings = settings
    _setup_exception_handlers(app)
    app.include_router(router)
    app.include_router(memory_router)
    return app
