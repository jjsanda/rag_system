"""Graceful-degradation wrappers.

Wrap a real (cloud/local) provider with a deterministic offline fallback. When the primary
is *unavailable* (``ProviderUnavailableError``: exhausted retries, unreachable host) the
wrapper transparently degrades to the fallback and logs a warning. Configuration errors
such as a bad API key surface as ``EmbeddingProviderError``/``LLMProviderError`` (HTTP 4xx)
and are **not** masked; those should be fixed, not hidden. Enabled by
``RAG_FALLBACK_TO_OFFLINE=true``.

The fallback embedder must be constructed with the same ``dimension`` as the primary so the
vectors remain compatible with the store (the factory wires this).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence

from rag.domain.exceptions import ProviderUnavailableError
from rag.domain.ports import LLM, Embedder

logger = logging.getLogger(__name__)

__all__ = ["FallbackLLM", "FallbackEmbedder"]


class FallbackLLM:
    """An :class:`~rag.domain.ports.LLM` that degrades to an offline LLM on unavailability."""

    def __init__(self, primary: LLM, fallback: LLM) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def model_name(self) -> str:
        return self._primary.model_name

    @property
    def provider(self) -> str:
        return self._primary.provider

    async def generate(self, *, system: str, prompt: str) -> str:
        try:
            return await self._primary.generate(system=system, prompt=prompt)
        except ProviderUnavailableError as exc:
            logger.warning(
                "LLM '%s' unavailable (%s); degrading to offline '%s'",
                self._primary.provider,
                exc,
                self._fallback.provider,
            )
            return await self._fallback.generate(system=system, prompt=prompt)

    async def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]:
        produced = False
        try:
            async for token in self._primary.stream(system=system, prompt=prompt):
                produced = True
                yield token
            return
        except ProviderUnavailableError as exc:
            if produced:
                raise  # cannot recover once tokens have been emitted
            logger.warning(
                "LLM '%s' unavailable (%s); degrading to offline '%s'",
                self._primary.provider,
                exc,
                self._fallback.provider,
            )
        async for token in self._fallback.stream(system=system, prompt=prompt):
            yield token


class FallbackEmbedder:
    """An :class:`~rag.domain.ports.Embedder` that degrades to an offline embedder."""

    def __init__(self, primary: Embedder, fallback: Embedder) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def model_name(self) -> str:
        return self._primary.model_name

    @property
    def dimension(self) -> int:
        return self._primary.dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            return await self._primary.embed(texts)
        except ProviderUnavailableError as exc:
            logger.warning(
                "embedder '%s' unavailable (%s); degrading to offline fallback",
                self._primary.model_name,
                exc,
            )
            return await self._fallback.embed(texts)
