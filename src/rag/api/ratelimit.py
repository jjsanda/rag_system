"""A small in-memory fixed-window rate limiter (pure ASGI).

Scoped to a path prefix (``/api/v1`` by default) so liveness/readiness probes and ``/metrics``
are never throttled, since Kubernetes polls the probes frequently. Keyed by
client IP. In-process only; for multi-replica deployments put a shared limiter (e.g. an API
gateway or Redis-backed limiter) in front. Disabled when the configured rate is empty.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque

from starlette.types import ASGIApp, Receive, Scope, Send

from rag.domain.exceptions import ConfigError

__all__ = ["RateLimitMiddleware", "parse_rate"]

_UNIT_SECONDS = {"second": 1.0, "minute": 60.0, "hour": 3600.0}


def parse_rate(spec: str) -> tuple[int, float]:
    """Parse a ``"<count>/<unit>"`` spec (e.g. ``"60/minute"``) into ``(count, window_seconds)``."""
    count_str, _, unit = spec.partition("/")
    unit = unit.strip().lower().rstrip("s") or "minute"
    try:
        count = int(count_str.strip())
        window = _UNIT_SECONDS[unit]
    except (ValueError, KeyError) as exc:
        raise ConfigError(f"invalid rate limit spec {spec!r} (expected e.g. '60/minute')") from exc
    return count, window


class RateLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        limit: int,
        window_s: float,
        path_prefix: str = "/api/v1",
    ) -> None:
        self.app = app
        self.limit = limit
        self.window = window_s
        self.path_prefix = path_prefix
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(self.path_prefix):
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        key = client[0] if client else "anonymous"
        now = time.monotonic()
        bucket = self._hits[key]
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()

        if len(bucket) >= self.limit:
            await self._reject(scope, send)
            return

        bucket.append(now)
        await self.app(scope, receive, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        request_id = scope.get("state", {}).get("request_id")
        payload = json.dumps(
            {
                "error": {"type": "RateLimitError", "message": "rate limit exceeded"},
                "request_id": request_id,
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"retry-after", str(int(self.window)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})
