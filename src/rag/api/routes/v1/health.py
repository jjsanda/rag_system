"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from rag.api.deps import ServicesDep
from rag.api.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: the process is up (no dependency checks)."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(services: ServicesDep) -> ReadyResponse:
    """Readiness: the vector store is reachable. Returns 503 when it is not."""
    store_ok = await services.store.healthy()
    if not store_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="vector store not ready"
        )
    return ReadyResponse(status="ok", degraded=False, checks={"vector_store": True})
