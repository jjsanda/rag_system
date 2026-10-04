"""Immutable domain models.

These are plain, framework-free :mod:`dataclasses` (``frozen=True, slots=True``) so the
core has no dependency on Pydantic, FastAPI or any provider SDK. Pydantic models live only
at the API boundary (:mod:`rag.api.schemas`).

Example:
    >>> doc = Document(id="d1", text="Bees build hexagonal comb.", title="Bees")
    >>> chunk = Chunk(id="d1::0", doc_id="d1", index=0, text=doc.text, token_count=6)
    >>> chunk.id
    'd1::0'
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Document",
    "Chunk",
    "EmbeddedChunk",
    "RetrievedChunk",
    "Source",
    "Timings",
    "Answer",
    "Query",
    "IngestionResult",
    "StreamEvent",
]


@dataclass(frozen=True, slots=True)
class Document:
    """An input document as loaded, before cleaning and chunking."""

    id: str
    text: str
    title: str | None = None
    source: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A token-bounded slice of a :class:`Document`."""

    id: str
    doc_id: str
    index: int
    text: str
    token_count: int
    title: str | None = None
    source: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A :class:`Chunk` paired with its embedding vector."""

    chunk: Chunk
    embedding: Sequence[float]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by a similarity search, with its relevance ``score``.

    ``score`` is normalised so that higher means more relevant (cosine similarity),
    regardless of the underlying store's native metric.
    """

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Source:
    """A numbered citation source presented to the model and returned to the caller.

    ``number`` is the 1-based marker used in the answer text as ``[number]``.
    """

    number: int
    doc_id: str
    chunk_id: str
    title: str | None
    score: float
    snippet: str


@dataclass(frozen=True, slots=True)
class Timings:
    """Latency breakdown for a single query (milliseconds).

    ``retrieval_ms`` covers query-embedding + vector search; ``generation_ms`` covers the
    LLM call. They are measured separately so they can be charted independently.
    """

    retrieval_ms: float
    generation_ms: float
    total_ms: float
    embedding_ms: float | None = None


@dataclass(frozen=True, slots=True)
class Answer:
    """The final grounded answer with its citations and latency breakdown."""

    text: str
    sources: tuple[Source, ...]
    citations: tuple[int, ...]
    timings: Timings
    model: str
    llm_provider: str


@dataclass(frozen=True, slots=True)
class Query:
    """A user query, with an optional override for how many chunks to retrieve."""

    text: str
    top_k: int = 4


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Counts and latency from an ingestion run."""

    documents: int
    chunks: int
    vectors: int
    took_ms: float


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """A single Server-Sent-Events payload emitted while streaming an answer.

    ``event`` is one of ``"meta"``, ``"token"``, ``"sources"``, ``"done"`` or ``"error"``;
    ``data`` is a JSON-serialisable mapping.
    """

    event: str
    data: Mapping[str, Any]
