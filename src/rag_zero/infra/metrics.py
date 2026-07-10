"""Prometheus metrics registry and helpers."""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

QUERY_TOTAL = Counter(
    "rag_zero_query_total",
    "Total number of /query requests",
    ["status"],
    registry=REGISTRY,
)

ABSTENTION_TOTAL = Counter(
    "rag_zero_abstention_total",
    "Total number of abstained answers",
    registry=REGISTRY,
)

HALLUCINATION_TOTAL = Counter(
    "rag_zero_hallucination_total",
    "Total number of hallucinated answers",
    registry=REGISTRY,
)

STAGE_LATENCY = Histogram(
    "rag_zero_stage_latency_seconds",
    "Per-stage latency in seconds",
    ["stage"],
    registry=REGISTRY,
)


def metrics_app() -> bytes:
    return generate_latest(REGISTRY)
