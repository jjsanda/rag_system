"""Pure-ASGI request-context middleware.

Implemented as raw ASGI (not ``BaseHTTPMiddleware``) for two reasons: it does not buffer
streaming responses (so SSE keeps streaming), and it runs the downstream app in the same
context, so the ``request_id`` bound here propagates into endpoint logs via contextvars.
It also assigns/echoes ``X-Request-ID`` and emits one structured access-log line per request.
"""

from __future__ import annotations

from time import perf_counter

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag.observability.logging import (
    bind_request_id,
    clear_request_context,
    get_logger,
    new_request_id,
)

__all__ = ["RequestContextMiddleware", "REQUEST_ID_HEADER"]

REQUEST_ID_HEADER = "x-request-id"
logger = get_logger("rag.api.access")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = Headers(scope=scope).get(REQUEST_ID_HEADER) or new_request_id()
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id
        bind_request_id(request_id)
        start = perf_counter()

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
                logger.info(
                    "request",
                    method=scope.get("method"),
                    path=scope.get("path"),
                    status=message["status"],
                    duration_ms=round((perf_counter() - start) * 1000, 2),
                )
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            clear_request_context()
