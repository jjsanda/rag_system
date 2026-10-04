"""In-memory caches: an LRU embedding cache and a TTL response cache."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from time import monotonic

from rag.domain.models import Answer

__all__ = ["LruEmbeddingCache", "InMemoryResponseCache"]


class LruEmbeddingCache:
    """Bounded LRU cache implementing the :class:`~rag.domain.ports.EmbeddingCache` port."""

    def __init__(self, max_size: int = 10_000) -> None:
        self._data: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max(1, max_size)

    def get(self, key: str) -> list[float] | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: str, value: Sequence[float]) -> None:
        self._data[key] = list(value)
        self._data.move_to_end(key)
        while len(self._data) > self._max_size:
            self._data.popitem(last=False)


class InMemoryResponseCache:
    """Optional response cache implementing :class:`~rag.domain.ports.ResponseCache`.

    A ``ttl_seconds`` of ``None`` keeps entries until eviction; a positive value expires
    them. Note: entries are not invalidated on re-ingestion, so enable this only when the
    corpus is stable (it is off by default).
    """

    def __init__(self, ttl_seconds: float | None = None) -> None:
        self._data: dict[str, tuple[float | None, Answer]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Answer | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, answer = entry
        if expires_at is not None and monotonic() > expires_at:
            self._data.pop(key, None)
            return None
        return answer

    def set(self, key: str, value: Answer) -> None:
        expires_at = monotonic() + self._ttl if self._ttl else None
        self._data[key] = (expires_at, value)
