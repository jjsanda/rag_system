"""Integration tests for the FastAPI app, driven over ASGI (no real server, no network)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from rag.api.main import create_app
from rag.config.settings import Settings
from rag.factory import Services, build_services

_RESET_KEYS = (
    "RAG_EMBEDDING_PROVIDER",
    "RAG_LLM_PROVIDER",
    "RAG_VECTOR_STORE",
    "RAG_API_KEY",
    "RAG_RATE_LIMIT",
)


def _make_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **env: str
) -> tuple[FastAPI, Services]:
    for key in _RESET_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAG_FAISS_PATH", str(tmp_path / "faiss"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    settings = Settings(_env_file=None)
    services = build_services(settings)
    return create_app(settings, services=services), services


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[AsyncClient]:
    app, services = _make_app(monkeypatch, tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
    await services.aclose()


async def _ingest_two_docs(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/ingest",
        json={
            "documents": [
                {
                    "id": "bees",
                    "title": "Bees",
                    "text": "A honeybee colony has one queen and many female workers.",
                },
                {
                    "id": "tcp",
                    "title": "TCP",
                    "text": "TCP is a connection oriented transport layer protocol.",
                },
            ]
        },
    )


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers.get("x-request-id")


async def test_ready(client: AsyncClient) -> None:
    response = await client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["vector_store"] is True


async def test_info(client: AsyncClient) -> None:
    body = (await client.get("/api/v1/info")).json()
    assert body["embedding_provider"] == "fake"
    assert body["llm_provider"] == "extractive"
    assert body["vector_store"] == "faiss"
    assert body["version"]


async def test_ingest_sample_corpus(client: AsyncClient) -> None:
    response = await client.post("/api/v1/ingest", json={"use_sample_corpus": True})
    assert response.status_code == 200
    body = response.json()
    assert body["documents"] >= 8
    assert body["chunks"] > 0
    assert body["vectors"] == body["chunks"]


async def test_ingest_requires_input(client: AsyncClient) -> None:
    response = await client.post("/api/v1/ingest", json={})
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "InputValidationError"


async def test_query_returns_grounded_cited_answer(client: AsyncClient) -> None:
    await _ingest_two_docs(client)
    response = await client.post(
        "/api/v1/query", json={"question": "queen and workers in a honeybee colony", "top_k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["citations"]
    assert body["sources"][0]["doc_id"] == "bees"
    assert body["llm_provider"] == "extractive"
    assert body["timings"]["total_ms"] >= 0
    assert body["request_id"]


async def test_query_stream_emits_sse_events(client: AsyncClient) -> None:
    await _ingest_two_docs(client)
    response = await client.post(
        "/api/v1/query/stream", json={"question": "honeybee colony queen", "top_k": 2}
    )
    assert response.status_code == 200
    text = response.text
    for marker in ("event: meta", "event: sources", "event: token", "event: done"):
        assert marker in text


async def test_query_rejects_oversized_question(client: AsyncClient) -> None:
    response = await client.post("/api/v1/query", json={"question": "x" * 5000})
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "InputValidationError"


async def test_metrics_endpoint(client: AsyncClient) -> None:
    await _ingest_two_docs(client)
    await client.post("/api/v1/query", json={"question": "honeybee"})
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "rag_queries_total" in body
    assert "rag_retrieval_seconds" in body


async def test_openapi_and_docs(client: AsyncClient) -> None:
    assert (await client.get("/openapi.json")).status_code == 200
    assert (await client.get("/docs")).status_code == 200


async def test_api_key_enforced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app, services = _make_app(monkeypatch, tmp_path, RAG_API_KEY="secret-key")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        assert (await http_client.get("/api/v1/info")).status_code == 401
        ok = await http_client.get("/api/v1/info", headers={"X-API-Key": "secret-key"})
        assert ok.status_code == 200
        # health is unauthenticated
        assert (await http_client.get("/health")).status_code == 200
    await services.aclose()


async def test_rate_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app, services = _make_app(monkeypatch, tmp_path, RAG_RATE_LIMIT="2/minute")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        statuses = [(await http_client.get("/api/v1/info")).status_code for _ in range(4)]
        # probes are exempt from rate limiting even after the API limit is exhausted
        probe = await http_client.get("/health")
    await services.aclose()
    assert statuses[:2] == [200, 200]
    assert 429 in statuses[2:]
    assert probe.status_code == 200
