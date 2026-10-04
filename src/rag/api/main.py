"""FastAPI application factory and composition with the rest of the system.

``create_app`` builds the app, configures logging/metrics/tracing, installs middleware and
exception handlers, and wires the routers. Services are built once in the lifespan (or
injected for tests). A module-level ``app = create_app()`` is exposed for
``uvicorn rag.api.main:app``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from rag import __version__
from rag.api.errors import register_exception_handlers
from rag.api.middleware import RequestContextMiddleware
from rag.api.ratelimit import RateLimitMiddleware, parse_rate
from rag.api.routes.v1 import health, ingest, query
from rag.config import get_settings
from rag.config.settings import Settings
from rag.factory import Services, build_services
from rag.observability.logging import configure_logging
from rag.observability.metrics import render_latest
from rag.observability.tracing import configure_tracing, instrument_fastapi

logger = logging.getLogger(__name__)

__all__ = ["create_app", "app"]


def create_app(settings: Settings | None = None, *, services: Services | None = None) -> FastAPI:
    """Build the FastAPI application. Pass ``services`` to inject a pre-built stack (tests)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    configure_tracing(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        built = services or build_services(settings)
        app.state.services = built
        logger.info("application startup complete")
        try:
            yield
        finally:
            await built.aclose()
            logger.info("application shutdown complete")

    # Interactive docs are served everywhere except prod, where the Swagger/ReDoc UIs are
    # hidden (the OpenAPI schema stays available for tooling). RAG_ENV drives this.
    docs_enabled = settings.env != "prod"
    app = FastAPI(
        title="RAG System API",
        version=__version__,
        summary="A provider-agnostic Retrieval-Augmented Generation service.",
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    # Pre-set services so ASGI-transport tests work without running the lifespan.
    if services is not None:
        app.state.services = services

    # Middleware (added last = outermost): request context wraps everything, so the
    # rate limiter's 429 still carries the request id and is access-logged.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if settings.rate_limit:
        limit, window = parse_rate(settings.rate_limit)
        app.add_middleware(RateLimitMiddleware, limit=limit, window_s=window)
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(ingest.router)

    if settings.metrics_enabled:

        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            payload, content_type = render_latest()
            return Response(content=payload, media_type=content_type)

    instrument_fastapi(app, settings)
    return app


app = create_app()
