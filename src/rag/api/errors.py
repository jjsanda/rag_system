"""Map the domain exception hierarchy onto clean JSON HTTP responses."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag.api.schemas import ErrorBody, ErrorResponse
from rag.domain.exceptions import (
    ConfigError,
    EmbeddingProviderError,
    IngestionError,
    InputValidationError,
    LLMProviderError,
    ProviderError,
    ProviderUnavailableError,
    RagError,
    RateLimitError,
    RetrievalError,
    VectorStoreError,
)
from rag.observability.metrics import PROVIDER_ERRORS

logger = logging.getLogger(__name__)

__all__ = ["register_exception_handlers"]

# Most-specific first: ProviderUnavailableError must precede ProviderError, etc.
_STATUS_BY_TYPE: tuple[tuple[type[RagError], int], ...] = (
    (InputValidationError, 422),
    (RateLimitError, 429),
    (ProviderUnavailableError, 503),
    (VectorStoreError, 502),
    (EmbeddingProviderError, 502),
    (LLMProviderError, 502),
    (ProviderError, 502),
    (IngestionError, 400),
    (RetrievalError, 500),
    (ConfigError, 500),
    (RagError, 500),
)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _status_for(exc: RagError) -> int:
    for exc_type, status_code in _STATUS_BY_TYPE:
        if isinstance(exc, exc_type):
            return status_code
    return 500  # pragma: no cover


async def _handle_rag_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RagError)
    status_code = _status_for(exc)
    if isinstance(exc, ProviderError) and exc.provider:
        PROVIDER_ERRORS.labels(provider=exc.provider).inc()
    if status_code >= 500:
        logger.error("request failed: %s", exc, extra={"error_type": type(exc).__name__})
    body = ErrorResponse(
        error=ErrorBody(type=type(exc).__name__, message=str(exc)),
        request_id=_request_id(request),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error")
    body = ErrorResponse(
        error=ErrorBody(type="InternalServerError", message="an unexpected error occurred"),
        request_id=_request_id(request),
    )
    return JSONResponse(status_code=500, content=body.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Register the domain-error and catch-all handlers on the app."""
    app.add_exception_handler(RagError, _handle_rag_error)
    app.add_exception_handler(Exception, _handle_unexpected)
