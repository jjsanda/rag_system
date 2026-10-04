"""Round-trip tests for the optional vector stores, run in-process (no server, no network).

Each test skips cleanly if the backend's extra is not installed.
"""

from __future__ import annotations

import pytest

from rag.domain.models import Chunk, EmbeddedChunk


def _embedded(chunk_id: str, vector: list[float], *, title: str) -> EmbeddedChunk:
    return EmbeddedChunk(Chunk(chunk_id, "d", 0, "bees make honey", 4, title=title), vector)


async def test_chroma_roundtrip() -> None:
    pytest.importorskip("chromadb")
    from rag.adapters.vectorstore.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(collection_name="test_chroma")
    await store.reset()
    await store.upsert(
        [
            _embedded("d::0", [1.0, 0.0, 0.0], title="Bees"),
            _embedded("d::1", [0.0, 1.0, 0.0], title="Other"),
        ]
    )
    hits = await store.search([1.0, 0.0, 0.0], top_k=1)
    assert hits[0].chunk.id == "d::0"
    assert hits[0].chunk.title == "Bees"
    assert hits[0].score > 0.9
    assert await store.count() == 2

    await store.reset()
    assert await store.count() == 0


async def test_qdrant_roundtrip() -> None:
    pytest.importorskip("qdrant_client")
    from rag.adapters.vectorstore.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore(dimension=3, url=":memory:", collection_name="test_qdrant")
    await store.upsert(
        [
            _embedded("d::0", [1.0, 0.0, 0.0], title="Bees"),
            _embedded("d::1", [0.0, 1.0, 0.0], title="Other"),
        ]
    )
    hits = await store.search([1.0, 0.0, 0.0], top_k=1)
    assert hits[0].chunk.id == "d::0"
    assert hits[0].chunk.title == "Bees"
    assert hits[0].score > 0.9
    assert await store.count() == 2

    await store.reset()
    assert await store.count() == 0
    await store.aclose()
