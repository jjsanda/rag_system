"""FastAPI dependency-injection helpers.

The composition root builds the :class:`Services` once at startup and stores them on
``app.state``; routes resolve what they need through these dependencies, and tests override
them to inject fakes.
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Header, HTTPException, Request, status

from rag.factory import Services

__all__ = ["get_services", "verify_api_key", "ServicesDep"]


def get_services(request: Request) -> Services:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="services not initialised"
        )
    return cast(Services, services)


ServicesDep = Annotated[Services, Depends(get_services)]


def verify_api_key(
    services: ServicesDep,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Enforce ``X-API-Key`` when ``RAG_API_KEY`` is configured (no-op otherwise)."""
    expected = services.settings.api_key
    if expected and x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key"
        )
