"""Chroma vector store adapter (local, embedded; optional ``chroma`` extra).

A zero-server alternative to FAISS that persists to disk and uses cosine space. Chroma's
client is synchronous, so calls run in a worker thread. Switched on by
``RAG_VECTOR_STORE=chroma``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from rag.domain.exceptions import ConfigError
from rag.domain.models import Chunk, EmbeddedChunk, RetrievedChunk

__all__ = ["ChromaVectorStore"]


class ChromaVectorStore:
    """Implements the :class:`~rag.domain.ports.VectorStore` port on top of Chroma."""

    def __init__(
        self,
        *,
        collection_name: str = "rag_documents",
        persist_path: str | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - only without the 'chroma' extra
            raise ConfigError("chromadb is not installed; install the 'chroma' extra") from exc
        settings = chromadb.config.Settings(anonymized_telemetry=False)
        self._client = (
            chromadb.PersistentClient(path=persist_path, settings=settings)
            if persist_path
            else chromadb.EphemeralClient(settings=settings)
        )
        self._name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    async def upsert(self, items: Sequence[EmbeddedChunk]) -> None:
        if not items:
            return
        await asyncio.to_thread(self._upsert, items)

    def _upsert(self, items: Sequence[EmbeddedChunk]) -> None:
        # Typed as `Any` so the code is identical whether or not the optional `chroma`
        # extra (with its strict embedding types) is installed.
        embeddings: Any = [list(item.embedding) for item in items]
        self._collection.upsert(
            ids=[item.chunk.id for item in items],
            embeddings=embeddings,
            documents=[item.chunk.text for item in items],
            metadatas=[_to_metadata(item.chunk) for item in items],
        )

    async def search(self, embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        if top_k < 1:
            return []
        result = await asyncio.to_thread(self._query, list(embedding), top_k)
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        hits: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=True):
            hits.append(
                RetrievedChunk(chunk=_from_metadata(metadata, text), score=1.0 - float(distance))
            )
        return hits

    def _query(self, embedding: list[float], top_k: int) -> Any:
        query_embeddings: Any = [embedding]
        return self._collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    async def count(self) -> int:
        return int(await asyncio.to_thread(self._collection.count))

    async def reset(self) -> None:
        await asyncio.to_thread(self._reset)

    def _reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    async def healthy(self) -> bool:
        try:
            await asyncio.to_thread(self._client.heartbeat)
            return True
        except Exception:  # pragma: no cover - backend-specific
            return False


def _to_metadata(chunk: Chunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id,
        "index": chunk.index,
        "token_count": chunk.token_count,
        "title": chunk.title or "",
        "source": chunk.source or "",
    }


def _from_metadata(metadata: dict[str, Any], text: str) -> Chunk:
    return Chunk(
        id=f"{metadata['doc_id']}::{metadata['index']}",
        doc_id=str(metadata["doc_id"]),
        index=int(metadata["index"]),
        text=text,
        token_count=int(metadata["token_count"]),
        title=metadata.get("title") or None,
        source=metadata.get("source") or None,
    )
