"""Application configuration."""

from __future__ import annotations

from functools import lru_cache

from rag.config.settings import Settings

__all__ = ["Settings", "get_settings"]


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` loaded from the environment."""
    return Settings()
