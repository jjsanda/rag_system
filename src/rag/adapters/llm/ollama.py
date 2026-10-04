"""Ollama chat LLM adapter (raw httpx, async, retried) for fully-local generation.

Talks to ``POST {base_url}/api/chat``. Non-streaming sends ``stream: false`` and reads a
single JSON object; streaming reads newline-delimited JSON (Ollama is JSONL, not SSE).
Switched on by ``RAG_LLM_PROVIDER=ollama`` (needs a running ``ollama serve`` + a pulled model).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from rag.adapters._http import post_json
from rag.domain.exceptions import LLMProviderError, ProviderUnavailableError

__all__ = ["OllamaChatLLM"]


class OllamaChatLLM:
    """Implements the :class:`~rag.domain.ports.LLM` port against a local Ollama server."""

    def __init__(
        self,
        *,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        self._model = model
        self._options = {"temperature": temperature, "num_predict": max_tokens}
        self._attempts = max_retries
        self._backoff = backoff_s
        self._headers = {"Content-Type": "application/json"}
        self._client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout_s))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "ollama"

    def _payload(self, system: str, prompt: str, *, stream: bool) -> dict[str, object]:
        return {
            "model": self._model,
            "stream": stream,
            "options": self._options,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }

    async def generate(self, *, system: str, prompt: str) -> str:
        data = await post_json(
            self._client,
            "/api/chat",
            headers=self._headers,
            payload=self._payload(system, prompt, stream=False),
            attempts=self._attempts,
            base_delay=self._backoff,
            provider=self.provider,
            error_cls=LLMProviderError,
        )
        message: str = data.get("message", {}).get("content", "")
        return message

    async def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]:
        payload = self._payload(system, prompt, stream=True)
        try:
            async with self._client.stream(
                "POST", "/api/chat", headers=self._headers, json=payload
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise LLMProviderError(
                        f"ollama returned HTTP {response.status_code}: {body[:200]}",
                        provider="ollama",
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    content = event.get("message", {}).get("content")
                    if content:
                        yield content
                    if event.get("done"):
                        break
        except httpx.TransportError as exc:
            raise ProviderUnavailableError(
                f"ollama is unavailable: {exc}", provider="ollama"
            ) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
