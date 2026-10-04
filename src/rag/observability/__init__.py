"""Observability: structured logging, Prometheus metrics and optional OpenTelemetry."""

from __future__ import annotations

from rag.observability.logging import (
    bind_request_id,
    clear_request_context,
    configure_logging,
    get_logger,
    new_request_id,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "new_request_id",
    "bind_request_id",
    "clear_request_context",
]
