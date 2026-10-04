"""Disk-backed embedding cache using ``diskcache`` (persists across restarts)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import diskcache

logger = logging.getLogger(__name__)

__all__ = ["DiskEmbeddingCache"]


class DiskEmbeddingCache:
    """Persistent embedding cache implementing the :class:`~rag.domain.ports.EmbeddingCache` port."""

    def __init__(self, path: str | Path, size_limit_bytes: int = 512 * 1024 * 1024) -> None:
        self._cache = diskcache.Cache(str(path), size_limit=size_limit_bytes)
        logger.debug("disk embedding cache at %s", path)

    def get(self, key: str) -> list[float] | None:
        value = self._cache.get(key)
        return list(value) if value is not None else None

    def set(self, key: str, value: Sequence[float]) -> None:
        self._cache.set(key, list(value))

    def close(self) -> None:
        self._cache.close()
