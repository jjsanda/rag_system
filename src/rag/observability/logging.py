"""Structured JSON logging with request-id propagation.

Bridges stdlib ``logging`` (used by the adapters) and ``structlog`` (used by the app)
through a single :class:`structlog.stdlib.ProcessorFormatter`, so every log line, wherever
it originates, renders as one consistent JSON object that includes any ``request_id`` bound
to the current context.
"""

from __future__ import annotations

import logging
import uuid

import structlog

__all__ = [
    "configure_logging",
    "get_logger",
    "new_request_id",
    "bind_request_id",
    "clear_request_context",
]


def configure_logging(level: str = "INFO", *, json_logs: bool = True) -> None:
    """Configure structlog + stdlib logging. Idempotent; safe to call at startup."""
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger (optionally named)."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def new_request_id() -> str:
    """Generate a short, unique request id."""
    return uuid.uuid4().hex


def bind_request_id(request_id: str) -> None:
    """Bind ``request_id`` to the current (async) context so logs include it."""
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    """Clear any context-local log bindings (call at the end of a request)."""
    structlog.contextvars.clear_contextvars()
