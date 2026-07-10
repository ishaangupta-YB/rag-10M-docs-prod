"""Compile the LangGraph CRAG agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from rag_zero.agent.nodes import AgentDeps, AgentNodes
from rag_zero.models.domain import AgentState


def compile_langgraph(deps: AgentDeps) -> Any:
    """Build and compile the agent graph."""
    nodes = AgentNodes(deps)

    graph = StateGraph(AgentState)
    graph.add_node("route", nodes.route)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade)
    graph.add_node("refine", nodes.refine)
    graph.add_node("generate", nodes.generate)
    graph.add_node("verify", nodes.verify)
    graph.add_node("finalize", nodes.finalize)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        nodes.route_decision,
        {"finalize": "finalize", "retrieve": "retrieve"},
    )
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        nodes.grade_decision,
        {"generate": "generate", "refine": "refine", "finalize": "finalize"},
    )
    graph.add_edge("refine", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
