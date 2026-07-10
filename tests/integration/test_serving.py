"""Integration tests for the FastAPI serving layer."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag_zero.serving.app import create_app


def test_health_returns_200() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_returns_service_info() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "rag-zero"


def test_metrics_returns_prometheus() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"rag_zero_query_total" in response.content
