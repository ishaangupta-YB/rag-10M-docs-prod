"""Business services for query, ingestion, and evaluation."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

from rag_zero.agent.generator import CitedGenerator
from rag_zero.agent.nodes import AgentDeps
from rag_zero.agent.policy import AbstentionPolicy
from rag_zero.agent.verifier import VerificationGate
from rag_zero.clients.embedder_client import OpenAIEmbedderClient
from rag_zero.clients.llm_client import LLMClient
from rag_zero.clients.mem0_client import Mem0Service
from rag_zero.clients.reranker_client import OpenAIRerankerClient
from rag_zero.clients.verifier_client import VerifierClient
from rag_zero.evaluation.metrics import Metrics
from rag_zero.ingestion.indexer import IndexStore
from rag_zero.models.domain import AgentState, FinalAnswer
from rag_zero.retrieval.hybrid import HybridRetriever

if TYPE_CHECKING:
    from rag_zero.clients.base import BaseLLMClient, BaseRerankerClient
    from rag_zero.config import Settings


def _build_llm(settings: Settings) -> LLMClient:
    return LLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )


def _build_embedder(settings: Settings) -> OpenAIEmbedderClient:
    return OpenAIEmbedderClient(
        base_url=settings.embedder_base_url,
        model=settings.embedder_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        max_retries=settings.embedder_max_retries,
        expected_dim=settings.embedder_dim,
        query_instruction=settings.embedder_query_instruction,
    )


def _build_reranker(settings: Settings, llm: BaseLLMClient) -> BaseRerankerClient:
    return OpenAIRerankerClient(llm)


def _build_verifier(settings: Settings, llm: BaseLLMClient) -> VerificationGate:
    return VerificationGate(VerifierClient(llm, tau=settings.tau_claim), tau=settings.tau_claim)


class _Job:
    def __init__(self) -> None:
        self.id = str(uuid.uuid4())
        self.status = "pending"
        self.progress = 0.0
        self.message = ""
        self.task: asyncio.Task[None] | None = None


class IngestionService:
    """Async ingestion job tracker and runner."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jobs: dict[str, _Job] = {}

    def get_job(self, job_id: str) -> _Job | None:
        return self._jobs.get(job_id)

    def ingest(self, source: str = "hotpotqa", slice_n: int | None = None) -> str:
        job = _Job()
        self._jobs[job.id] = job
        job.task = asyncio.create_task(self._run(job, source, slice_n))
        return job.id

    async def _run(self, job: _Job, source: str, slice_n: int | None) -> None:
        from rag_zero.cli import ingest

        job.status = "running"
        job.progress = 0.1
        job.message = "starting ingestion"

        def _blocking_ingest() -> None:
            # Typer commands are sync; run in a worker thread.
            ingest(source=source, slice=slice_n)

        try:
            await asyncio.to_thread(_blocking_ingest)
            job.status = "completed"
            job.progress = 1.0
            job.message = "ingestion complete"
        except Exception as exc:
            job.status = "failed"
            job.message = str(exc)


class EvaluationService:
    """Run golden-set evaluation and return metrics."""

    def __init__(self, settings: Settings, query_service: QueryService | None = None) -> None:
        self.settings = settings
        self.query_service = query_service

    async def evaluate(self) -> dict[str, Any]:
        metrics = Metrics(self.settings, query_service=self.query_service)
        return await metrics.run_golden_set()


class QueryService:
    """Run the CRAG agent for a single question."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = _build_llm(settings)
        self.embedder = _build_embedder(settings)
        self.reranker = _build_reranker(settings, self.llm)
        self.verifier = _build_verifier(settings, self.llm)
        self.index_store = IndexStore(settings, self.embedder)
        self.retriever = HybridRetriever(
            settings, self.index_store, self.embedder, self.reranker
        )
        self.generator = CitedGenerator(self.llm, max_tokens=settings.llm_max_tokens)
        self.policy = AbstentionPolicy(tau_abstain=settings.tau_abstain)
        self.memory_service = Mem0Service(settings)

    async def query(
        self, question: str, *, user_id: str | None = None, use_memory: bool = True
    ) -> FinalAnswer:
        from rag_zero.agent.graph import compile_langgraph

        deps = AgentDeps(
            llm_client=self.llm,
            retriever=self.retriever,
            generator=self.generator,
            verifier=self.verifier,
            policy=self.policy,
            settings=self.settings,
        )
        graph = compile_langgraph(deps)
        memory_context: list[str] = []
        effective_user = user_id or self.settings.mem0_user_id
        if (
            use_memory
            and self.settings.enable_memory
            and effective_user
            and self.memory_service.is_available()
        ):
            memory_context = await self.memory_service.search(
                question, user_id=effective_user
            )

        initial_state: AgentState = {
            "question": question,
            "route": "",
            "query": "",
            "evidence": [],
            "grade": 0.0,
            "draft": "",
            "gate": "",
            "final": None,
            "hops": 0,
            "latencies": {},
            "memory_context": memory_context,
        }
        start = time.perf_counter()
        result = await graph.ainvoke(initial_state)
        total = time.perf_counter() - start
        final_answer = result.get("final")
        if not isinstance(final_answer, FinalAnswer):
            final_answer = FinalAnswer(
                status="abstained",
                answer="Unable to produce a final answer.",
                citations=[],
                min_support=0.0,
                reason="graph_failed",
                latencies=result.get("latencies", {}),
                hops=result.get("hops", 0),
            )
        final_answer.latencies["total"] = total

        if (
            use_memory
            and self.settings.enable_memory
            and effective_user
            and self.memory_service.is_available()
        ):
            await self.memory_service.add(
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": final_answer.answer},
                ],
                user_id=effective_user,
                metadata={
                    "status": final_answer.status,
                    "citations": final_answer.citations,
                    "min_support": final_answer.min_support,
                    "reason": final_answer.reason,
                },
            )

        return final_answer

    async def close(self) -> None:
        await self.llm.close()
        if self.embedder is not None:
            await self.embedder.close()
