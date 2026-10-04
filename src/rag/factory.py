"""Composition root: the single place that turns :class:`Settings` into wired services.

This is where the hexagon is assembled: concrete adapters are selected by configuration and
injected into the framework-free :class:`IngestionService` / :class:`QueryService`. Nothing
else in the codebase constructs adapters, so swapping a provider is a config change here, not
a code change anywhere else.

Wrapping order for a real (cloud/local) provider:
``Fallback(Caching(provider), offline)`` for embeddings and ``Fallback(provider, extractive)``
for the LLM, so a cache miss still hits the real provider, and an *unavailable* real
provider degrades to the deterministic offline one.

Example:
    >>> from rag.config.settings import Settings
    >>> services = build_services(Settings(_env_file=None))  # offline defaults
    >>> services.llm.provider, services.embedder.model_name
    ('extractive', 'hashing-384')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, assert_never, cast, runtime_checkable

from rag.adapters.cache.disk import DiskEmbeddingCache
from rag.adapters.cache.memory import InMemoryResponseCache, LruEmbeddingCache
from rag.adapters.embeddings.caching import CachingEmbedder
from rag.adapters.embeddings.fake import HashingEmbedder
from rag.adapters.embeddings.local import SentenceTransformerEmbedder
from rag.adapters.embeddings.openai import OpenAIEmbedder
from rag.adapters.fallback import FallbackEmbedder, FallbackLLM
from rag.adapters.llm.anthropic import AnthropicChatLLM
from rag.adapters.llm.extractive import ExtractiveLLM
from rag.adapters.llm.ollama import OllamaChatLLM
from rag.adapters.llm.openai import OpenAIChatLLM
from rag.adapters.loaders.filesystem import FilesystemLoader
from rag.adapters.tokenizer import TiktokenTokenizer
from rag.adapters.vectorstore.chroma_store import ChromaVectorStore
from rag.adapters.vectorstore.faiss_store import FaissVectorStore
from rag.adapters.vectorstore.qdrant_store import QdrantVectorStore
from rag.config.settings import Settings
from rag.domain.chunking import TokenAwareChunker
from rag.domain.cleaning import DefaultTextCleaner
from rag.domain.exceptions import ConfigError
from rag.domain.pipeline import IngestionService, QueryService
from rag.domain.ports import (
    LLM,
    DocumentLoader,
    Embedder,
    EmbeddingCache,
    ResponseCache,
    Tokenizer,
    VectorStore,
)
from rag.domain.prompt import build_system_prompt

logger = logging.getLogger(__name__)

__all__ = ["Services", "build_services"]


@runtime_checkable
class _Closeable(Protocol):
    async def aclose(self) -> None: ...


@dataclass(slots=True)
class Services:
    """The fully-wired application surface handed to the API and CLI."""

    settings: Settings
    tokenizer: Tokenizer
    embedder: Embedder
    store: VectorStore
    llm: LLM
    ingestion: IngestionService
    query: QueryService
    loader: DocumentLoader
    closeables: list[_Closeable] = field(default_factory=list)

    async def aclose(self) -> None:
        """Close any provider clients holding network resources (best-effort)."""
        for closeable in self.closeables:
            try:
                await closeable.aclose()
            except Exception as exc:  # pragma: no cover
                logger.warning("error closing %s: %s", type(closeable).__name__, exc)


def build_services(settings: Settings) -> Services:
    """Build and wire every adapter and service from configuration."""
    closeables: list[_Closeable] = []

    tokenizer = TiktokenTokenizer(settings.tokenizer_encoding)
    embedder = _wrap_embedder(_build_base_embedder(settings, closeables), settings)
    store = _build_store(settings, embedder.dimension, closeables)
    llm = _wrap_llm(_build_base_llm(settings, closeables), settings)

    chunker = TokenAwareChunker(
        tokenizer, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    loader = FilesystemLoader()
    ingestion = IngestionService(
        cleaner=DefaultTextCleaner(),
        chunker=chunker,
        embedder=embedder,
        store=store,
        loader=loader,
    )
    query = QueryService(
        embedder=embedder,
        store=store,
        llm=llm,
        system_prompt=build_system_prompt(settings.system_prompt),
        default_top_k=settings.top_k,
        snippet_chars=240,
        response_cache=_build_response_cache(settings),
    )
    logger.info(
        "services built: embedding=%s llm=%s store=%s",
        settings.embedding_provider,
        settings.llm_provider,
        settings.vector_store,
    )
    return Services(
        settings=settings,
        tokenizer=tokenizer,
        embedder=embedder,
        store=store,
        llm=llm,
        ingestion=ingestion,
        query=query,
        loader=loader,
        closeables=closeables,
    )


# -- embeddings -------------------------------------------------------------------------


def _build_base_embedder(settings: Settings, closeables: list[_Closeable]) -> Embedder:
    provider = settings.embedding_provider
    if provider == "fake":
        return HashingEmbedder(dim=settings.embedding_dim)
    if provider == "local":
        return SentenceTransformerEmbedder(settings.local_embedding_model)
    if provider == "openai":
        if not settings.openai_api_key:
            raise ConfigError("RAG_EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        embedder = OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            base_url=settings.openai_base_url or "https://api.openai.com",
            default_dim=settings.embedding_dim,
            timeout_s=settings.request_timeout_s,
            max_retries=settings.provider_max_retries,
            backoff_s=settings.provider_backoff_s,
        )
        closeables.append(embedder)
        return embedder
    assert_never(provider)


def _wrap_embedder(base: Embedder, settings: Settings) -> Embedder:
    cache = _build_embedding_cache(settings)
    embedder: Embedder = CachingEmbedder(base, cache) if cache is not None else base
    if settings.embedding_provider != "fake" and settings.fallback_to_offline:
        embedder = FallbackEmbedder(embedder, HashingEmbedder(dim=base.dimension))
    return embedder


def _build_embedding_cache(settings: Settings) -> EmbeddingCache | None:
    if settings.embedding_cache == "memory":
        return LruEmbeddingCache(max_size=settings.embedding_cache_size)
    if settings.embedding_cache == "disk":
        return DiskEmbeddingCache(settings.embedding_cache_path)
    return None


# -- LLM --------------------------------------------------------------------------------


def _build_base_llm(settings: Settings, closeables: list[_Closeable]) -> LLM:
    provider = settings.llm_provider
    if provider == "extractive":
        return ExtractiveLLM()
    if provider == "ollama":
        llm: LLM = OllamaChatLLM(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_s=max(settings.request_timeout_s, 60.0),
            max_retries=settings.provider_max_retries,
            backoff_s=settings.provider_backoff_s,
        )
    elif provider == "openai":
        if not settings.openai_api_key:
            raise ConfigError("RAG_LLM_PROVIDER=openai requires OPENAI_API_KEY")
        llm = OpenAIChatLLM(
            api_key=settings.openai_api_key,
            model=settings.openai_llm_model,
            base_url=settings.openai_base_url or "https://api.openai.com",
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.request_timeout_s,
            max_retries=settings.provider_max_retries,
            backoff_s=settings.provider_backoff_s,
        )
    elif provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigError("RAG_LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        llm = AnthropicChatLLM(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_llm_model,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.request_timeout_s,
            max_retries=settings.provider_max_retries,
            backoff_s=settings.provider_backoff_s,
        )
    else:
        assert_never(provider)
    closeables.append(cast(_Closeable, llm))  # real LLM adapters own an httpx client to close
    return llm


def _wrap_llm(base: LLM, settings: Settings) -> LLM:
    if settings.llm_provider != "extractive" and settings.fallback_to_offline:
        return FallbackLLM(base, ExtractiveLLM())
    return base


# -- vector store -----------------------------------------------------------------------


def _build_store(settings: Settings, dimension: int, closeables: list[_Closeable]) -> VectorStore:
    kind = settings.vector_store
    if kind == "faiss":
        return FaissVectorStore(
            dimension, collection_name=settings.collection_name, persist_path=settings.faiss_path
        )
    if kind == "chroma":
        return ChromaVectorStore(
            collection_name=settings.collection_name, persist_path=settings.chroma_path
        )
    if kind == "qdrant":
        store = QdrantVectorStore(
            dimension=dimension,
            collection_name=settings.collection_name,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        closeables.append(store)
        return store
    assert_never(kind)


# -- response cache ---------------------------------------------------------------------


def _build_response_cache(settings: Settings) -> ResponseCache | None:
    if settings.response_cache == "memory":
        return InMemoryResponseCache(ttl_seconds=settings.response_cache_ttl_s)
    return None
