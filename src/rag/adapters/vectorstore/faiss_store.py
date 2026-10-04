"""FAISS-backed vector store: the local default, with no separate service to run.

Vectors live in a flat ``IndexFlatIP``; because we L2-normalise every vector, inner product
equals cosine similarity. Chunk metadata is kept alongside in memory and optionally
persisted to disk as JSON so an ingested corpus survives restarts. Great as a
dependency-free default; switch to Qdrant for production scale (same port, config-only).

Example:
    >>> import asyncio
    >>> from rag.domain.models import Chunk, EmbeddedChunk
    >>> store = FaissVectorStore(dimension=3)
    >>> chunk = Chunk("d::0", "d", 0, "hello", 1)
    >>> asyncio.run(store.upsert([EmbeddedChunk(chunk, [0.0, 1.0, 0.0])]))
    >>> hits = asyncio.run(store.search([0.0, 2.0, 0.0], top_k=1))
    >>> hits[0].chunk.id, round(hits[0].score, 3)
    ('d::0', 1.0)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag.domain.exceptions import VectorStoreError
from rag.domain.models import Chunk, EmbeddedChunk, RetrievedChunk

logger = logging.getLogger(__name__)

__all__ = ["FaissVectorStore"]


class FaissVectorStore:
    """Implements the :class:`~rag.domain.ports.VectorStore` port with an in-process FAISS index."""

    def __init__(
        self,
        dimension: int,
        *,
        collection_name: str = "rag_documents",
        persist_path: str | Path | None = None,
    ) -> None:
        self._dim = dimension
        self._collection = collection_name
        self._path = Path(persist_path) if persist_path else None
        self._ids: list[str] = []
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._index = faiss.IndexFlatIP(dimension)
        if self._path is not None:
            self._load()

    async def upsert(self, items: Sequence[EmbeddedChunk]) -> None:
        for item in items:
            self._chunks[item.chunk.id] = item.chunk
            self._vectors[item.chunk.id] = self._as_vector(item.embedding)
        self._rebuild()
        self._save()

    async def search(self, embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        if not self._ids or top_k < 1:
            return []
        query = self._as_vector(embedding).reshape(1, -1)
        scores, positions = self._index.search(query, min(top_k, len(self._ids)))
        results: list[RetrievedChunk] = []
        for score, position in zip(scores[0].tolist(), positions[0].tolist(), strict=True):
            if position < 0:
                continue
            chunk = self._chunks[self._ids[position]]
            results.append(RetrievedChunk(chunk=chunk, score=float(score)))
        return results

    async def count(self) -> int:
        return len(self._ids)

    async def reset(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        self._rebuild()
        self._save()

    async def healthy(self) -> bool:
        return True

    # -- internals ----------------------------------------------------------------------

    def _as_vector(self, embedding: Sequence[float]) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).copy()
        if vector.ndim != 1 or vector.shape[0] != self._dim:
            raise VectorStoreError(
                f"expected a {self._dim}-d vector, got shape {vector.shape}", provider="faiss"
            )
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector

    def _rebuild(self) -> None:
        self._ids = list(self._vectors.keys())
        self._index = faiss.IndexFlatIP(self._dim)
        if self._ids:
            matrix = np.vstack([self._vectors[cid] for cid in self._ids]).astype(np.float32)
            self._index.add(matrix)

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.mkdir(parents=True, exist_ok=True)
        payload = {
            "dimension": self._dim,
            "collection": self._collection,
            "items": [
                {"chunk": _chunk_to_dict(self._chunks[cid]), "vector": self._vectors[cid].tolist()}
                for cid in self._ids
            ],
        }
        (self._path / "store.json").write_text(json.dumps(payload), encoding="utf-8")

    def _load(self) -> None:
        if self._path is None:
            return
        store_file = self._path / "store.json"
        if not store_file.exists():
            return
        try:
            payload = json.loads(store_file.read_text(encoding="utf-8"))
            for entry in payload.get("items", []):
                chunk = _chunk_from_dict(entry["chunk"])
                self._chunks[chunk.id] = chunk
                self._vectors[chunk.id] = np.asarray(entry["vector"], dtype=np.float32)
            self._rebuild()
            logger.info("loaded %d vectors from %s", len(self._ids), store_file)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("could not load FAISS store from %s: %s", store_file, exc)


def _chunk_to_dict(chunk: Chunk) -> dict[str, object]:
    return {
        "id": chunk.id,
        "doc_id": chunk.doc_id,
        "index": chunk.index,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "title": chunk.title,
        "source": chunk.source,
        "metadata": dict(chunk.metadata),
    }


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    return Chunk(
        id=str(data["id"]),
        doc_id=str(data["doc_id"]),
        index=int(data["index"]),
        text=str(data["text"]),
        token_count=int(data["token_count"]),
        title=data.get("title"),
        source=data.get("source"),
        metadata=data.get("metadata") or {},
    )
