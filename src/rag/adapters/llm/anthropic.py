"""Anthropic Messages API LLM adapter (raw httpx, async, retried).

Talks to ``POST {base_url}/v1/messages`` with the ``x-api-key`` + ``anthropic-version``
headers. ``temperature`` is deliberately **not** sent: current Claude models (Opus 4.8/4.7,
Sonnet 4.6) reject sampling parameters with a 400, and the cheaper default models do not
need it. Switched on by ``RAG_LLM_PROVIDER=anthropic``; requires ``ANTHROPIC_API_KEY``.
Streaming parses ``content_block_delta`` events for ``text_delta`` chunks.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from rag.adapters._http import post_json
from rag.domain.exceptions import LLMProviderError, ProviderUnavailableError

__all__ = ["AnthropicChatLLM"]


class AnthropicChatLLM:
    """Implements the :class:`~rag.domain.ports.LLM` port against the Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-haiku-4-5",
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 512,
        anthropic_version: str = "2023-06-01",
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._attempts = max_retries
        self._backoff = backoff_s
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": anthropic_version,
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout_s))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "anthropic"

    def _payload(self, system: str, prompt: str, *, stream: bool) -> dict[str, object]:
        return {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "stream": stream,
            "messages": [{"role": "user", "content": prompt}],
        }

    async def generate(self, *, system: str, prompt: str) -> str:
        data = await post_json(
            self._client,
            "/v1/messages",
            headers=self._headers,
            payload=self._payload(system, prompt, stream=False),
            attempts=self._attempts,
            base_delay=self._backoff,
            provider=self.provider,
            error_cls=LLMProviderError,
        )
        blocks = data.get("content") or []
        return "".join(block.get("text", "") for block in blocks if block.get("type") == "text")

    async def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]:
        payload = self._payload(system, prompt, stream=True)
        try:
            async with self._client.stream(
                "POST", "/v1/messages", headers=self._headers, json=payload
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise LLMProviderError(
                        f"anthropic returned HTTP {response.status_code}: {body[:200]}",
                        provider="anthropic",
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[len("data:") :].strip())
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield delta["text"]
        except httpx.TransportError as exc:
            raise ProviderUnavailableError(
                f"anthropic is unavailable: {exc}", provider="anthropic"
            ) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
