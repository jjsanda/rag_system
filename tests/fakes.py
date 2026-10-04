"""Lightweight in-test fakes implementing the domain ports.

These keep the *domain* unit tests free of any real adapter or heavy dependency. The
production deterministic adapters (``fake`` embedder, ``extractive`` LLM, FAISS store) are
implemented and tested separately under :mod:`rag.adapters`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import AsyncIterator, Sequence

from rag.domain.exceptions import EmbeddingProviderError, LLMProviderError
from rag.domain.models import Answer, Document, EmbeddedChunk, RetrievedChunk
from rag.domain.prompt import parse_context


class CharTokenizer:
    """A character-level tokenizer that round-trips exactly (one token == one char)."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(chr(t) for t in tokens)

    def count(self, text: str) -> int:
        return len(text)


class FakeEmbedder:
    """Deterministic bag-of-words embedder: shared vocabulary => higher cosine similarity."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "fake-bow"

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for word in text.lower().split():
            bucket = int(hashlib.sha1(word.encode("utf-8")).hexdigest(), 16) % self._dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class FailingEmbedder(FakeEmbedder):
    """An embedder that always raises, to exercise error paths."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingProviderError("embedder unavailable", provider="test")


class InMemoryStore:
    """A tiny cosine-similarity vector store backed by a Python list."""

    def __init__(self) -> None:
        self._items: list[EmbeddedChunk] = []

    async def upsert(self, items: Sequence[EmbeddedChunk]) -> None:
        incoming = {item.chunk.id for item in items}
        self._items = [e for e in self._items if e.chunk.id not in incoming]
        self._items.extend(items)

    async def search(self, embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(chunk=item.chunk, score=_cosine(embedding, item.embedding))
            for item in self._items
        ]
        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored[:top_k]

    async def count(self) -> int:
        return len(self._items)

    async def reset(self) -> None:
        self._items.clear()

    async def healthy(self) -> bool:
        return True


class FakeLLM:
    """Echoes the top context block with a ``[1]`` citation; counts generate() calls."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-llm"

    @property
    def provider(self) -> str:
        return "fake"

    async def generate(self, *, system: str, prompt: str) -> str:
        self.calls += 1
        blocks = parse_context(prompt)
        if not blocks:
            return "I do not have enough information to answer."
        return f"{blocks[0].text} [1]"

    async def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]:
        text = await self.generate(system=system, prompt=prompt)
        for word in text.split(" "):
            yield word + " "


class StreamFailingLLM(FakeLLM):
    """Yields one token, then fails mid-generation (exercises the stream error path)."""

    async def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]:
        yield "partial "
        raise LLMProviderError("stream interrupted", provider="test")


class StaticLoader:
    """A document loader that returns a fixed set of documents regardless of source."""

    def __init__(self, docs: Sequence[Document]) -> None:
        self._docs = list(docs)

    def load(self, source: str | object) -> list[Document]:
        return list(self._docs)


class MemoryResponseCache:
    """A trivial dict-backed response cache."""

    def __init__(self) -> None:
        self._store: dict[str, Answer] = {}

    def get(self, key: str) -> Answer | None:
        return self._store.get(key)

    def set(self, key: str, value: Answer) -> None:
        self._store[key] = value


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
