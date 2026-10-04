"""Qdrant vector store adapter (production option; optional ``qdrant`` extra).

Uses the async Qdrant client and cosine distance. Chunk ids are mapped to deterministic
UUID point ids (Qdrant requires int/UUID ids) with the original id kept in the payload.
Switched on by ``RAG_VECTOR_STORE=qdrant`` (runs as a Docker/Helm service; ``location=":memory:"``
is used by tests). The collection is created lazily on first use.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from rag.domain.exceptions import ConfigError, VectorStoreError
from rag.domain.models import Chunk, EmbeddedChunk, RetrievedChunk

__all__ = ["QdrantVectorStore"]

# Stable namespace so a chunk id always maps to the same Qdrant point id.
_NAMESPACE = uuid.UUID("6a1f0c2e-0000-4000-8000-00000000d0c5")


class QdrantVectorStore:
    """Implements the :class:`~rag.domain.ports.VectorStore` port against Qdrant."""

    def __init__(
        self,
        *,
        dimension: int,
        collection_name: str = "rag_documents",
        url: str = "http://localhost:6333",
        api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import AsyncQdrantClient, models
        except ImportError as exc:  # pragma: no cover - only without the 'qdrant' extra
            raise ConfigError("qdrant-client is not installed; install the 'qdrant' extra") from exc
        self._models = models
        self._dim = dimension
        self._name = collection_name
        if url.startswith("http"):
            self._client = AsyncQdrantClient(url=url, api_key=api_key)
        else:  # e.g. ":memory:" or a local path
            self._client = AsyncQdrantClient(location=url)
        self._ready = False

    async def _ensure(self) -> None:
        if self._ready:
            return
        if not await self._client.collection_exists(self._name):
            await self._client.create_collection(
                collection_name=self._name,
                vectors_config=self._models.VectorParams(
                    size=self._dim, distance=self._models.Distance.COSINE
                ),
            )
        self._ready = True

    async def upsert(self, items: Sequence[EmbeddedChunk]) -> None:
        if not items:
            return
        await self._ensure()
        points = [
            self._models.PointStruct(
                id=_point_id(item.chunk.id),
                vector=list(item.embedding),
                payload=_payload(item.chunk),
            )
            for item in items
        ]
        await self._client.upsert(collection_name=self._name, points=points)

    async def search(self, embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        if top_k < 1:
            return []
        await self._ensure()
        response = await self._client.query_points(
            collection_name=self._name, query=list(embedding), limit=top_k, with_payload=True
        )
        return [
            RetrievedChunk(chunk=_from_payload(point.payload), score=float(point.score))
            for point in response.points
        ]

    async def count(self) -> int:
        await self._ensure()
        return int((await self._client.count(collection_name=self._name)).count)

    async def reset(self) -> None:
        if await self._client.collection_exists(self._name):
            await self._client.delete_collection(self._name)
        self._ready = False
        await self._ensure()

    async def healthy(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:  # pragma: no cover - backend-specific
            return False

    async def aclose(self) -> None:
        await self._client.close()


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def _payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "doc_id": chunk.doc_id,
        "index": chunk.index,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "title": chunk.title,
        "source": chunk.source,
    }


def _from_payload(payload: dict[str, Any] | None) -> Chunk:
    if payload is None:
        raise VectorStoreError("qdrant returned a point without a payload", provider="qdrant")
    return Chunk(
        id=str(payload["chunk_id"]),
        doc_id=str(payload["doc_id"]),
        index=int(payload["index"]),
        text=str(payload["text"]),
        token_count=int(payload["token_count"]),
        title=payload.get("title"),
        source=payload.get("source"),
    )
