"""OpenAI embeddings adapter (raw httpx, async, retried, batched).

Talks to ``POST {base_url}/v1/embeddings``. The vector ``dimension`` is resolved from a
static map of known models (so the vector store can be sized before any network call);
unknown models fall back to a configurable default. Switched on by
``RAG_EMBEDDING_PROVIDER=openai``; requires ``OPENAI_API_KEY``.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from rag.adapters._http import post_json
from rag.domain.exceptions import EmbeddingProviderError

__all__ = ["OpenAIEmbedder"]

# Native output dimensions for current OpenAI embedding models.
_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    """Implements the :class:`~rag.domain.ports.Embedder` port against the OpenAI Embeddings API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com",
        default_dim: int = 1536,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        self._model = model
        self._dim = _MODEL_DIMS.get(model, default_dim)
        self._attempts = max_retries
        self._backoff = backoff_s
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout_s))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        data = await post_json(
            self._client,
            "/v1/embeddings",
            headers=self._headers,
            payload={"model": self._model, "input": list(texts)},
            attempts=self._attempts,
            base_delay=self._backoff,
            provider="openai",
            error_cls=EmbeddingProviderError,
        )
        rows = data.get("data") or []
        if len(rows) != len(texts):
            raise EmbeddingProviderError(
                f"openai returned {len(rows)} embeddings for {len(texts)} inputs", provider="openai"
            )
        # Preserve input order explicitly via the per-row index.
        ordered = sorted(rows, key=lambda row: row["index"])
        return [list(row["embedding"]) for row in ordered]

    async def aclose(self) -> None:
        await self._client.aclose()
