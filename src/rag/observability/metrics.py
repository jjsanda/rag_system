"""Prometheus metrics for the RAG service.

Retrieval and generation are timed separately (the two dominant, very differently-shaped
latencies in a RAG request) alongside request/ingest/error counters. Exposed at ``/metrics``.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from rag.domain.models import Timings

__all__ = [
    "RETRIEVAL_SECONDS",
    "GENERATION_SECONDS",
    "EMBEDDING_SECONDS",
    "QUERIES",
    "INGESTED_DOCUMENTS",
    "INGESTED_CHUNKS",
    "PROVIDER_ERRORS",
    "INJECTION_FLAGS",
    "observe_query",
    "render_latest",
]

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

RETRIEVAL_SECONDS = Histogram(
    "rag_retrieval_seconds", "Query embedding + vector search latency", buckets=_LATENCY_BUCKETS
)
GENERATION_SECONDS = Histogram(
    "rag_generation_seconds", "LLM answer-generation latency", buckets=_LATENCY_BUCKETS
)
EMBEDDING_SECONDS = Histogram(
    "rag_embedding_seconds", "Query-embedding latency", buckets=_LATENCY_BUCKETS
)
QUERIES = Counter("rag_queries_total", "Queries handled, by outcome", ["status"])
INGESTED_DOCUMENTS = Counter("rag_ingested_documents_total", "Documents ingested")
INGESTED_CHUNKS = Counter("rag_ingested_chunks_total", "Chunks produced during ingestion")
PROVIDER_ERRORS = Counter(
    "rag_provider_errors_total", "Provider failures, by provider", ["provider"]
)
INJECTION_FLAGS = Counter(
    "rag_prompt_injection_flags_total", "Queries flagged for prompt-injection patterns"
)


def observe_query(timings: Timings) -> None:
    """Record the per-phase latencies of a completed query (milliseconds -> seconds)."""
    RETRIEVAL_SECONDS.observe(timings.retrieval_ms / 1000.0)
    GENERATION_SECONDS.observe(timings.generation_ms / 1000.0)
    if timings.embedding_ms is not None:
        EMBEDDING_SECONDS.observe(timings.embedding_ms / 1000.0)


def render_latest() -> tuple[bytes, str]:
    """Return the current metrics payload and its Prometheus content type."""
    return generate_latest(), CONTENT_TYPE_LATEST
