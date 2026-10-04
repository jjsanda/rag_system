"""Domain exception hierarchy.

All errors raised by the core and its adapters derive from :class:`RagError`, which lets
the API layer map a small, stable set of types onto HTTP responses without leaking
framework or provider details into the domain.

Example:
    >>> try:
    ...     raise EmbeddingProviderError("rate limited", provider="openai")
    ... except ProviderError as exc:
    ...     print(exc.provider, "->", exc)
    openai -> rate limited
"""

from __future__ import annotations

__all__ = [
    "RagError",
    "ConfigError",
    "InputValidationError",
    "IngestionError",
    "RetrievalError",
    "ProviderError",
    "EmbeddingProviderError",
    "LLMProviderError",
    "VectorStoreError",
    "ProviderUnavailableError",
    "RateLimitError",
]


class RagError(Exception):
    """Base class for every error raised by the RAG system."""


class ConfigError(RagError):
    """Configuration is missing or inconsistent (e.g. a provider lacks its API key)."""


class InputValidationError(RagError):
    """Caller-supplied input failed validation (too large, empty, malformed)."""


class IngestionError(RagError):
    """A document could not be loaded, cleaned, chunked or stored."""


class RetrievalError(RagError):
    """Retrieval from the vector store failed."""


class ProviderError(RagError):
    """Base class for failures originating in an external provider/adapter.

    Carries the originating ``provider`` name so it can be logged and surfaced
    without inspecting the concrete exception type.
    """

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class EmbeddingProviderError(ProviderError):
    """An embedding provider call failed."""


class LLMProviderError(ProviderError):
    """An LLM provider call failed."""


class VectorStoreError(ProviderError):
    """A vector-store backend call failed."""


class ProviderUnavailableError(ProviderError):
    """A provider is unreachable or unconfigured after retries were exhausted.

    The API maps this to ``503 Service Unavailable`` (used by graceful degradation).
    """


class RateLimitError(RagError):
    """The caller exceeded the configured request rate (mapped to HTTP 429)."""
