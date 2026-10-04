"""Deterministic hashing embedder: the offline default.

Uses signed feature hashing: each whitespace token is hashed to a bucket with a +/- sign
and accumulated, then the vector is L2-normalised. No model download, no network, perfectly
reproducible. Because cosine similarity tracks lexical overlap, retrieval over a
keyword-rich corpus stays meaningful: good enough to demo the full pipeline and to make
the evaluation metrics (hit@k / MRR) non-trivial without any external dependency.

Example:
    >>> import asyncio
    >>> emb = HashingEmbedder(dim=32)
    >>> vectors = asyncio.run(emb.embed(["honey bees build comb", "tcp is a protocol"]))
    >>> len(vectors) == 2 and len(vectors[0]) == 32
    True
    >>> asyncio.run(emb.embed(["repeat"])) == asyncio.run(emb.embed(["repeat"]))  # deterministic
    True
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from rag.domain.exceptions import ConfigError

__all__ = ["HashingEmbedder"]


class HashingEmbedder:
    """Implements the :class:`~rag.domain.ports.Embedder` port with no external model."""

    def __init__(self, dim: int = 384) -> None:
        if dim < 1:
            raise ConfigError(f"embedding dim must be >= 1, got {dim}")
        self._dim = dim

    @property
    def model_name(self) -> str:
        return f"hashing-{self._dim}"

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.tolist()
