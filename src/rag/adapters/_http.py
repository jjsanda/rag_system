"""Shared async HTTP plumbing for cloud provider adapters.

Centralises timeout + retry-with-exponential-backoff (tenacity) and maps transport
failures / retryable HTTP statuses onto the domain exception hierarchy:

* exhausted retries / unreachable host -> :class:`ProviderUnavailableError` (HTTP 503)
* a non-retryable ``4xx`` -> the provider's :class:`ProviderError` subclass (HTTP 502)

Streaming requests are intentionally *not* retried here (a partially-emitted stream cannot
be safely replayed); adapters handle their own stream error mapping.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag.domain.exceptions import ProviderError, ProviderUnavailableError

logger = logging.getLogger(__name__)

__all__ = ["RETRYABLE_STATUS", "acall_with_retry", "post_json"]

T = TypeVar("T")

# Transient HTTP statuses worth retrying (rate limits, gateway/overload errors).
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class _RetryableStatusError(Exception):
    """Internal marker that a response carried a retryable status code."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body[:200]}")
        self.status_code = status_code


# httpx.TimeoutException is a subclass of TransportError; both are retryable.
_RETRYABLE_EXC: tuple[type[BaseException], ...] = (_RetryableStatusError, httpx.TransportError)


async def acall_with_retry(
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay: float,
    retry_on: tuple[type[BaseException], ...] = _RETRYABLE_EXC,
) -> T:
    """Await ``func`` with exponential-backoff retries on transient failures."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max(1, attempts)),
        wait=wait_exponential(multiplier=base_delay, max=10.0),
        retry=retry_if_exception_type(retry_on),
        reraise=True,
    ):
        with attempt:
            return await func()
    raise RuntimeError("unreachable")  # pragma: no cover


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    attempts: int,
    base_delay: float,
    provider: str,
    error_cls: type[ProviderError],
) -> dict[str, Any]:
    """POST JSON with retries and return the parsed response, mapping failures to domain errors."""

    async def _do() -> httpx.Response:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code in RETRYABLE_STATUS:
            raise _RetryableStatusError(response.status_code, response.text)
        return response

    try:
        response = await acall_with_retry(_do, attempts=attempts, base_delay=base_delay)
    except (_RetryableStatusError, httpx.TransportError) as exc:
        raise ProviderUnavailableError(
            f"{provider} is unavailable: {exc}", provider=provider
        ) from exc

    if response.status_code >= 400:
        raise error_cls(
            f"{provider} returned HTTP {response.status_code}: {response.text[:200]}",
            provider=provider,
        )

    parsed: dict[str, Any] = response.json()
    return parsed
