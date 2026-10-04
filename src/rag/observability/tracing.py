"""Optional OpenTelemetry tracing (gated by ``OTEL_ENABLED``; needs the ``otel`` extra).

Kept entirely opt-in and lazily imported so the default install carries no OpenTelemetry
dependency and the import never fails when the extra is absent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rag.config.settings import Settings

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

__all__ = ["configure_tracing", "instrument_fastapi"]


def configure_tracing(settings: Settings) -> Any | None:
    """Set up an OTLP tracer provider if tracing is enabled and the SDK is installed."""
    if not settings.otel_enabled:
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("OTEL_ENABLED is set but opentelemetry is not installed (the 'otel' extra)")
        return None

    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry tracing enabled (service=%s)", settings.otel_service_name)
    return provider


def instrument_fastapi(app: FastAPI, settings: Settings) -> None:
    """Instrument the FastAPI app for tracing when enabled and available."""
    if not settings.otel_enabled:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:  # pragma: no cover - only without the 'otel' extra
        return
    FastAPIInstrumentor.instrument_app(app)
