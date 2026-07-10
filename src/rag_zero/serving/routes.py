"""FastAPI route handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST

from rag_zero import __version__
from rag_zero.infra.metrics import (
    ABSTENTION_TOTAL,
    HALLUCINATION_TOTAL,
    QUERY_TOTAL,
    STAGE_LATENCY,
    metrics_app,
)
from rag_zero.serving.schemas import (
    EvaluateResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
)

if TYPE_CHECKING:
    from rag_zero.config import Settings
    from rag_zero.serving.service import EvaluationService, IngestionService, QueryService

router = APIRouter()


def get_settings(request: Request) -> Settings:
    return cast("Settings", request.app.state.settings)


def get_query_service(request: Request) -> QueryService:
    return cast("QueryService", request.app.state.query_service)


def get_ingestion_service(request: Request) -> IngestionService:
    return cast("IngestionService", request.app.state.ingestion_service)


def get_evaluation_service(request: Request) -> EvaluationService:
    return cast("EvaluationService", request.app.state.evaluation_service)


@router.get("/health", response_model=HealthResponse)
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/health/ready")
def readiness(
    service: QueryService = Depends(get_query_service),
) -> Response:
    try:
        ok = service.index_store.exists()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"not ready: {exc}") from exc
    if not ok:
        raise HTTPException(status_code=503, detail="search index not built")
    return Response(content=b'{"status":"ready"}', media_type="application/json")


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=metrics_app(), media_type=CONTENT_TYPE_LATEST)


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    answer = await service.query(
        req.question,
        user_id=req.user_id,
        use_memory=req.use_memory,
    )
    QUERY_TOTAL.labels(status=answer.status).inc()
    if answer.status == "abstained":
        ABSTENTION_TOTAL.inc()
    for stage, latency in answer.latencies.items():
        STAGE_LATENCY.labels(stage=stage).observe(latency)
    return QueryResponse(**answer.dict())


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    req: IngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    job_id = service.ingest(source=req.source, slice_n=req.slice)
    return IngestResponse(job_id=job_id, status="accepted")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(
    job_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> JobStatusResponse:
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.message,
    )


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluateResponse:
    result = await service.evaluate()
    if result.get("hallucination_rate", 0.0) > 0.05:
        HALLUCINATION_TOTAL.inc()
    return EvaluateResponse(
        metrics=result.get("metrics", {}),
        confusion=result.get("confusion", {}),
        details=result.get("details", []),
    )


@router.get("/")
async def root() -> dict[str, Any]:
    return {"service": "rag-zero", "version": __version__}
