"""Unit tests for the CRAG LangGraph agent with fully mocked dependencies."""

from __future__ import annotations

import pytest

from rag_zero.agent.generator import CitedGenerator
from rag_zero.agent.nodes import AgentDeps
from rag_zero.agent.policy import AbstentionPolicy
from rag_zero.agent.verifier import VerificationGate
from rag_zero.clients.base import BaseLLMClient
from rag_zero.clients.verifier_client import VerifierClient
from rag_zero.config import Settings
from rag_zero.models.domain import RetrievedChunk


class _FakeLLM(BaseLLMClient):
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses

    async def chat(
        self,
        messages: list,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        content = " ".join([getattr(m, "content", "") for m in messages])
        for key, value in self.responses.items():
            if key in content:
                return value
        return ""

    async def json_prompt(self, messages: list, max_tokens: int | None = None) -> dict:
        raise NotImplementedError


class _FakeRetriever:
    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="c1",
                passage_id="p1",
                title="",
                text="Paris is the capital of France.",
                score=0.9,
                retrieval_method="fake",
            )
        ]

    def recall_at_k(self, evidence, gold_titles, k=None):
        return 1.0


class _FakeGenerator(BaseLLMClient):
    async def chat(
        self,
        messages: list,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        return "Paris is the capital of France [c1]."

    async def json_prompt(self, *args, **kwargs) -> dict:
        return {}


class _FakeJudgeLLM(BaseLLMClient):
    async def chat(
        self,
        messages: list,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        return "0.95"

    async def json_prompt(self, *args, **kwargs) -> dict:
        return {}


def _build_graph(route: str = "RETRIEVAL", grade: str = "0.95"):
    settings = Settings()
    llm = _FakeLLM(
        {
            "Label:": route,
            "Score": grade,
        }
    )
    deps = AgentDeps(
        llm_client=llm,
        retriever=_FakeRetriever(),  # type: ignore[arg-type]
        generator=CitedGenerator(_FakeGenerator()),  # type: ignore[arg-type]
        verifier=VerificationGate(VerifierClient(_FakeJudgeLLM()), tau=0.7),
        policy=AbstentionPolicy(),
        settings=settings,
    )
    from rag_zero.agent.graph import compile_langgraph

    return compile_langgraph(deps)


@pytest.mark.asyncio
async def test_retrieval_question_returns_answered() -> None:
    graph = _build_graph(route="RETRIEVAL", grade="0.95")
    result = await graph.ainvoke(
        {
            "question": "What is the capital of France?",
            "route": "",
            "query": "",
            "evidence": [],
            "grade": 0.0,
            "draft": "",
            "gate": "",
            "final": None,
            "hops": 0,
            "latencies": {},
            "memory_context": [],
        }
    )
    final = result["final"]
    assert final is not None
    assert final.status == "answered"
    assert "Paris" in final.answer


@pytest.mark.asyncio
async def test_direct_question_finalizes_immediately() -> None:
    graph = _build_graph(route="DIRECT", grade="0.0")
    result = await graph.ainvoke(
        {
            "question": "Hello",
            "route": "",
            "query": "",
            "evidence": [],
            "grade": 0.0,
            "draft": "",
            "gate": "",
            "final": None,
            "hops": 0,
            "latencies": {},
            "memory_context": [],
        }
    )
    final = result["final"]
    assert final is not None
    assert final.status == "abstained"
