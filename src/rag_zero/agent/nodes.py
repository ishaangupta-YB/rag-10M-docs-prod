"""LangGraph node implementations for the CRAG agent."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rag_zero.models.domain import AgentState, CitedAnswer
from rag_zero.models.prompts import (
    DECOMPOSER_PROMPT,
    FINAL_QUERY_PROMPT,
    GRADER_PROMPT,
    ROUTER_PROMPT,
)

if TYPE_CHECKING:
    from rag_zero.agent.generator import CitedGenerator
    from rag_zero.agent.policy import AbstentionPolicy
    from rag_zero.agent.verifier import VerificationGate
    from rag_zero.clients.base import BaseLLMClient
    from rag_zero.config import Settings
    from rag_zero.retrieval.hybrid import HybridRetriever


@dataclass(frozen=True)
class AgentDeps:
    """Injectable dependencies for all agent nodes."""

    llm_client: BaseLLMClient
    retriever: HybridRetriever
    generator: CitedGenerator
    verifier: VerificationGate
    policy: AbstentionPolicy
    settings: Settings


def _extract_score(text: str) -> float:
    numbers = [float(x) for x in re.findall(r"0?\.\d+|1\.0|0|1", text)]
    if numbers:
        return max(0.0, min(1.0, numbers[-1]))
    return 0.0


class AgentNodes:
    """Stateful node implementations bound to AgentDeps."""

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def _with_latency(
        self, state: AgentState, key: str
    ) -> Any:
        """Context manager style not directly used; see node methods."""
        return state

    async def route(self, state: AgentState) -> dict[str, Any]:
        start = time.perf_counter()
        memory = state.get("memory_context", [])
        memory_block = ""
        if memory:
            memory_lines = "\n".join(f"- {m}" for m in memory)
            memory_block = f"Relevant context from previous conversations:\n{memory_lines}\n\n"
        prompt = ROUTER_PROMPT.format(memory=memory_block, question=state["question"])
        label = (await self.deps.llm_client.complete(prompt, max_tokens=16)).strip().upper()
        route = "RETRIEVAL" if "RETRIEVAL" in label else "DIRECT"
        query = ""
        if route == "RETRIEVAL":
            prompt = DECOMPOSER_PROMPT.format(memory=memory_block, question=state["question"])
            query = (await self.deps.llm_client.complete(prompt, max_tokens=128)).strip()
        latency = time.perf_counter() - start
        return {
            "route": route,
            "query": query,
            "latencies": {**state.get("latencies", {}), "route": latency},
        }

    async def retrieve(self, state: AgentState) -> dict[str, Any]:
        start = time.perf_counter()
        query = state["query"] or state["question"]
        evidence = await self.deps.retriever.retrieve(query)
        latency = time.perf_counter() - start
        return {
            "evidence": evidence,
            "hops": state.get("hops", 0) + 1,
            "latencies": {**state.get("latencies", {}), "retrieve": latency},
        }

    async def grade(self, state: AgentState) -> dict[str, Any]:
        start = time.perf_counter()
        evidence = state.get("evidence", [])
        evidence_text = "\n\n".join(f"[{c.chunk_id}] {c.title}\n{c.text}" for c in evidence)
        prompt = GRADER_PROMPT.format(
            question=state["question"],
            evidence=evidence_text,
        )
        response = await self.deps.llm_client.complete(prompt, max_tokens=32)
        grade = _extract_score(response)
        latency = time.perf_counter() - start
        return {
            "grade": grade,
            "latencies": {**state.get("latencies", {}), "grade": latency},
        }

    async def refine(self, state: AgentState) -> dict[str, Any]:
        start = time.perf_counter()
        evidence = state.get("evidence", [])
        evidence_text = "\n".join(c.text[:500] for c in evidence)
        prompt = FINAL_QUERY_PROMPT.format(
            question=state["question"],
            query=state["query"],
            evidence=evidence_text,
        )
        query = (await self.deps.llm_client.complete(prompt, max_tokens=128)).strip()
        latency = time.perf_counter() - start
        return {
            "query": query,
            "latencies": {**state.get("latencies", {}), "refine": latency},
        }

    async def generate(self, state: AgentState) -> dict[str, Any]:
        start = time.perf_counter()
        cited = await self.deps.generator.generate(
            state["question"], state.get("evidence", [])
        )
        latency = time.perf_counter() - start
        return {
            "draft": cited.answer,
            "latencies": {**state.get("latencies", {}), "generate": latency},
        }

    async def verify(self, state: AgentState) -> dict[str, Any]:
        start = time.perf_counter()
        evidence = state.get("evidence", [])
        answer = CitedAnswer(
            answer=state["draft"],
            citations=list(set(re.findall(r"\[([A-Za-z0-9_:\-]+)\]", state["draft"]))),
            method="generate",
        )
        # Attach citations from CitedGenerator if it produced any.
        # But we recompute from draft for safety.
        status, _min_support, mean_support = await self.deps.verifier.check(
            answer, evidence
        )
        latency = time.perf_counter() - start
        return {
            "gate": status,
            "latencies": {
                **state.get("latencies", {}),
                "verify": latency,
                "mean_support": mean_support,
            },
        }

    async def finalize(self, state: AgentState) -> dict[str, Any]:
        start = time.perf_counter()
        evidence = state.get("evidence", [])
        answer = CitedAnswer(
            answer=state["draft"],
            citations=list(set(re.findall(r"\[([A-Za-z0-9_:\-]+)\]", state["draft"]))),
            method="generate",
        )
        status, min_support, _mean_support = await self.deps.verifier.check(
            answer, evidence
        )
        final = self.deps.policy.decide(
            question=state["question"],
            answer=answer,
            status=status,
            min_support=min_support,
            hops=state.get("hops", 0),
            latencies=state.get("latencies", {}),
        )
        latency = time.perf_counter() - start
        final.latencies["finalize"] = latency
        return {"final": final}

    def route_decision(self, state: AgentState) -> str:
        if state["route"] == "DIRECT":
            return "finalize"
        return "retrieve"

    def grade_decision(self, state: AgentState) -> str:
        grade = state.get("grade", 0.0)
        hops = state.get("hops", 0)
        if grade >= self.deps.settings.crag_ok:
            return "generate"
        if grade < self.deps.settings.crag_bad or hops >= self.deps.settings.max_hops:
            return "finalize"
        return "refine"
