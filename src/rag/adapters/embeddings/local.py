"""Local embedding adapter backed by sentence-transformers (no network at query time).

The model is loaded once; encoding runs in a worker thread (``asyncio.to_thread``) so it
never blocks the event loop. ``sentence-transformers`` is an optional dependency (the
``local`` extra) and is imported lazily. Switched on by ``RAG_EMBEDDING_PROVIDER=local``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from rag.domain.exceptions import ConfigError

__all__ = ["SentenceTransformerEmbedder"]


class SentenceTransformerEmbedder:
    """Implements the :class:`~rag.domain.ports.Embedder` port with a local ST model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - only without the 'local' extra
            raise ConfigError(
                "sentence-transformers is not installed; install the 'local' extra"
            ) from exc
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._normalize = normalize
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, list(texts))

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=self._normalize, convert_to_numpy=True
        )
        return [vector.tolist() for vector in vectors]
