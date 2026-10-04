"""Ports: the abstract boundaries of the hexagon.

Every external capability the core needs (tokenisation, embeddings, generation, vector
storage, loading, cleaning, caching) is expressed here as a :class:`typing.Protocol`.
The domain depends only on these structural interfaces; concrete adapters live under
:mod:`rag.adapters` and are wired in by :mod:`rag.factory`.

I/O-bound ports (embeddings, generation, vector store) are ``async`` so the API can stay
fully asynchronous end-to-end.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from rag.domain.models import (
    Answer,
    Document,
    EmbeddedChunk,
    RetrievedChunk,
)

__all__ = [
    "Tokenizer",
    "Embedder",
    "LLM",
    "VectorStore",
    "DocumentLoader",
    "TextCleaner",
    "EmbeddingCache",
    "ResponseCache",
]


@runtime_checkable
class Tokenizer(Protocol):
    """Token codec used for token-aware chunking (provider-independent)."""

    def encode(self, text: str) -> list[int]:
        """Return the token ids for ``text``."""

    def decode(self, tokens: Sequence[int]) -> str:
        """Inverse of :meth:`encode`."""

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""


@runtime_checkable
class Embedder(Protocol):
    """Maps text to dense vectors. Implementations should embed in batches."""

    @property
    def model_name(self) -> str:
        """Identifier of the underlying embedding model (for logs and responses)."""

    @property
    def dimension(self) -> int:
        """Dimensionality of the produced vectors."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving input order."""


@runtime_checkable
class LLM(Protocol):
    """A chat/completion model that answers from a grounded prompt."""

    @property
    def model_name(self) -> str:
        """Identifier of the underlying model."""

    @property
    def provider(self) -> str:
        """Short provider name, e.g. ``"openai"``, ``"anthropic"``, ``"extractive"``."""

    async def generate(self, *, system: str, prompt: str) -> str:
        """Return a full completion for the given system + user prompt."""

    def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]:
        """Yield the completion incrementally as text deltas."""


@runtime_checkable
class VectorStore(Protocol):
    """Persists embedded chunks and answers nearest-neighbour queries."""

    async def upsert(self, items: Sequence[EmbeddedChunk]) -> None:
        """Insert or update embedded chunks (idempotent on chunk id)."""

    async def search(self, embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks, highest score first."""

    async def count(self) -> int:
        """Return the number of stored vectors."""

    async def reset(self) -> None:
        """Remove all stored vectors (used by tests and re-ingestion)."""

    async def healthy(self) -> bool:
        """Return ``True`` if the backend is reachable (used by ``/ready``)."""


class DocumentLoader(Protocol):
    """Loads raw documents from a source (filesystem path, URL, ...)."""

    def load(self, source: str | Path) -> list[Document]:
        """Load and return documents from ``source``."""


class TextCleaner(Protocol):
    """Normalises raw document text prior to chunking."""

    def clean(self, text: str) -> str:
        """Return a cleaned copy of ``text``."""


class EmbeddingCache(Protocol):
    """Caches embedding vectors keyed by a content hash."""

    def get(self, key: str) -> list[float] | None:
        """Return the cached vector for ``key`` or ``None`` on a miss."""

    def set(self, key: str, value: Sequence[float]) -> None:
        """Store ``value`` under ``key``."""


class ResponseCache(Protocol):
    """Caches full answers keyed by a query hash (optional optimisation)."""

    def get(self, key: str) -> Answer | None:
        """Return the cached answer for ``key`` or ``None`` on a miss."""

    def set(self, key: str, value: Answer) -> None:
        """Store ``value`` under ``key``."""
