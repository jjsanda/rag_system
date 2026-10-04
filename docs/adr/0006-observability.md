# 6. Observability: structured logs, Prometheus metrics, optional tracing

- **Status:** Accepted
- **Date:** 2026-06-29

## Context

A service that calls embedding and LLM providers needs operational visibility: correlated
logs, latency that distinguishes retrieval from generation, and error/throughput counters,
without coupling the domain core to any telemetry library.

## Decision

- **Logging:** `structlog` rendering JSON, bridged with stdlib `logging` so app and adapter
  logs share one format; a `request_id` contextvar is merged into every line.
- **Metrics:** `prometheus-client` at `/metrics`, with separate histograms for
  `rag_retrieval_seconds`, `rag_generation_seconds`, `rag_embedding_seconds`, plus
  query/ingest/provider-error/injection counters. The domain pipeline *returns* timings; the
  API records them, so the core never imports Prometheus.
- **Tracing:** optional OpenTelemetry (OTLP), lazily imported and gated by `OTEL_ENABLED`.

## Consequences

- Retrieval and generation latency are charted separately, the split that matters most when
  tuning RAG.
- Telemetry stays at the edges; the core remains framework-free and easy to test.
- Tracing adds no dependency unless the `otel` extra is installed and enabled.
