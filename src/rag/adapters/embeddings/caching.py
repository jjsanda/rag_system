"""A caching decorator around any :class:`~rag.domain.ports.Embedder`.

Wraps an inner embedder and an :class:`~rag.domain.ports.EmbeddingCache`, serving cached
vectors per text and only embedding the cache misses (preserving input order). Embedding is
typically the most expensive step, so this materially speeds up repeated ingestion/queries.

Example:
    >>> import asyncio
    >>> from rag.adapters.embeddings.fake import HashingEmbedder
    >>> from rag.adapters.cache.memory import LruEmbeddingCache
    >>> cached = CachingEmbedder(HashingEmbedder(dim=16), LruEmbeddingCache())
    >>> a = asyncio.run(cached.embed(["x"]))
    >>> b = asyncio.run(cached.embed(["x"]))  # served from cache
    >>> a == b
    True
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from rag.domain.ports import Embedder, EmbeddingCache

__all__ = ["CachingEmbedder", "embedding_cache_key"]


def embedding_cache_key(model: str, text: str) -> str:
    """Stable cache key for a ``(model, text)`` pair."""
    return hashlib.sha256(f"{model}\x1f{text}".encode()).hexdigest()


class CachingEmbedder:
    """Implements the :class:`~rag.domain.ports.Embedder` port with read-through caching."""

    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(embedding_cache_key(self._inner.model_name, text))
            if cached is not None:
                results[i] = cached
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        if missing_texts:
            fresh = await self._inner.embed(missing_texts)
            for offset, original_index in enumerate(missing_indices):
                vector = fresh[offset]
                results[original_index] = vector
                self._cache.set(
                    embedding_cache_key(self._inner.model_name, missing_texts[offset]), vector
                )

        return [vector for vector in results if vector is not None]
