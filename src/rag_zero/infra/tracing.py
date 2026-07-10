"""OpenTelemetry setup for distributed tracing."""

from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from rag_zero.config import Settings  # noqa: TC001


def configure_tracing(settings: Settings) -> trace.Tracer | None:
    """Configure OTel tracer if endpoint is set."""
    if not settings.enable_tracing or not settings.otel_endpoint:
        return None

    resource = Resource.create({"service.name": "rag-zero"})
    provider = TracerProvider(resource=resource)

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception:
        # Fallback to no-op if exporter is unavailable.
        pass

    trace.set_tracer_provider(provider)
    return trace.get_tracer("rag-zero")


def span(tracer: trace.Tracer | None, name: str) -> Any:
    """Start a span if a tracer is available, otherwise return a no-op context."""
    from contextlib import nullcontext

    if tracer is None:
        return nullcontext()
    return tracer.start_as_current_span(name)
