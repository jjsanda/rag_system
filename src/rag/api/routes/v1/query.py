"""Query endpoints: synchronous answer, SSE streaming, and effective-config info."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from rag import __version__
from rag.api.deps import ServicesDep, verify_api_key
from rag.api.schemas import InfoResponse, QueryRequest, QueryResponse
from rag.config.settings import Settings
from rag.domain.exceptions import InputValidationError, RagError
from rag.domain.models import Query, Timings
from rag.domain.prompt import looks_like_injection
from rag.observability.metrics import INJECTION_FLAGS, QUERIES, observe_query

router = APIRouter(prefix="/api/v1", tags=["query"], dependencies=[Depends(verify_api_key)])


def _validate_question(question: str, settings: Settings) -> None:
    if len(question) > settings.max_question_chars:
        raise InputValidationError(
            f"question exceeds the maximum of {settings.max_question_chars} characters"
        )
    if looks_like_injection(question):
        INJECTION_FLAGS.inc()  # flag for observability; mitigation is structural (see prompt.py)


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, payload: QueryRequest, services: ServicesDep) -> QueryResponse:
    settings = services.settings
    _validate_question(payload.question, settings)
    user_query = Query(text=payload.question, top_k=payload.top_k or settings.top_k)
    try:
        answer = await services.query.answer(user_query)
    except RagError:
        QUERIES.labels(status="error").inc()
        raise
    QUERIES.labels(status="ok").inc()
    observe_query(answer.timings)
    return QueryResponse.from_answer(answer, getattr(request.state, "request_id", None))


@router.post("/query/stream")
async def query_stream(
    request: Request, payload: QueryRequest, services: ServicesDep
) -> EventSourceResponse:
    settings = services.settings
    _validate_question(payload.question, settings)
    user_query = Query(text=payload.question, top_k=payload.top_k or settings.top_k)

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        async for event in services.query.stream(user_query):
            if event.event == "done":
                timing = event.data["timings"]
                observe_query(
                    Timings(
                        retrieval_ms=timing["retrieval_ms"],
                        generation_ms=timing["generation_ms"],
                        total_ms=timing["total_ms"],
                        embedding_ms=timing.get("embedding_ms"),
                    )
                )
                QUERIES.labels(status="ok").inc()
            elif event.event == "error":
                QUERIES.labels(status="error").inc()
            yield {"event": event.event, "data": json.dumps(event.data)}

    return EventSourceResponse(event_generator())


@router.get("/info", response_model=InfoResponse)
async def info(services: ServicesDep) -> InfoResponse:
    return InfoResponse.model_validate({"version": __version__, **services.settings.public_info()})
