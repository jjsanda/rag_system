"""OpenAI chat-completions LLM adapter (raw httpx, async, retried).

Talks to ``POST {base_url}/v1/chat/completions`` directly so tests can mock it at the HTTP
layer with ``respx`` and to keep dependencies small (no SDK). Switched on by
``RAG_LLM_PROVIDER=openai``; requires ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from rag.adapters._http import post_json
from rag.domain.exceptions import LLMProviderError, ProviderUnavailableError

__all__ = ["OpenAIChatLLM"]


class OpenAIChatLLM:
    """Implements the :class:`~rag.domain.ports.LLM` port against the OpenAI Chat API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._attempts = max_retries
        self._backoff = backoff_s
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout_s))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openai"

    def _payload(self, system: str, prompt: str, *, stream: bool) -> dict[str, object]:
        return {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }

    async def generate(self, *, system: str, prompt: str) -> str:
        data = await post_json(
            self._client,
            "/v1/chat/completions",
            headers=self._headers,
            payload=self._payload(system, prompt, stream=False),
            attempts=self._attempts,
            base_delay=self._backoff,
            provider=self.provider,
            error_cls=LLMProviderError,
        )
        choices = data.get("choices") or []
        if not choices:
            raise LLMProviderError("openai returned no choices", provider="openai")
        content: str = choices[0]["message"]["content"] or ""
        return content

    async def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]:
        payload = self._payload(system, prompt, stream=True)
        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", headers=self._headers, json=payload
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise LLMProviderError(
                        f"openai returned HTTP {response.status_code}: {body[:200]}",
                        provider="openai",
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[len("data:") :].strip()
                    if chunk == "[DONE]":
                        break
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
        except httpx.TransportError as exc:
            raise ProviderUnavailableError(
                f"openai is unavailable: {exc}", provider="openai"
            ) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
