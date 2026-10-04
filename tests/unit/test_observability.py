"""Unit tests for logging configuration and Prometheus metrics."""

from __future__ import annotations

from rag.domain.models import Timings
from rag.observability.logging import (
    bind_request_id,
    clear_request_context,
    configure_logging,
    get_logger,
    new_request_id,
)
from rag.observability.metrics import QUERIES, observe_query, render_latest


def test_logging_helpers_run_without_error() -> None:
    configure_logging(json_logs=True)
    configure_logging(json_logs=False)  # exercise the console-renderer branch too
    assert len(new_request_id()) == 32
    assert new_request_id() != new_request_id()

    bind_request_id("req-123")
    get_logger("test").info("hello", custom=1)  # must not raise
    clear_request_context()


def test_metrics_render_includes_series() -> None:
    observe_query(Timings(retrieval_ms=12.0, generation_ms=34.0, total_ms=46.0, embedding_ms=5.0))
    QUERIES.labels(status="ok").inc()

    payload, content_type = render_latest()
    text = payload.decode()
    assert "rag_retrieval_seconds" in text
    assert "rag_generation_seconds" in text
    assert "rag_embedding_seconds" in text
    assert "rag_queries_total" in text
    assert "text/plain" in content_type
