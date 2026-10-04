"""Integration tests for the composition root (factory.build_services)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.adapters.embeddings.caching import CachingEmbedder
from rag.adapters.fallback import FallbackEmbedder, FallbackLLM
from rag.adapters.llm.extractive import ExtractiveLLM
from rag.adapters.llm.openai import OpenAIChatLLM
from rag.adapters.vectorstore.faiss_store import FaissVectorStore
from rag.config.settings import Settings
from rag.domain.exceptions import ConfigError
from rag.domain.models import Query
from rag.factory import build_services


def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **env: str) -> Settings:
    # Start from a clean slate so the CI/global RAG_* env does not leak into assertions.
    for key in (
        "RAG_EMBEDDING_PROVIDER",
        "RAG_LLM_PROVIDER",
        "RAG_VECTOR_STORE",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "RAG_FALLBACK_TO_OFFLINE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAG_FAISS_PATH", str(tmp_path / "faiss"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


async def test_offline_stack_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    services = build_services(_settings(monkeypatch, tmp_path))
    assert services.llm.provider == "extractive"
    assert isinstance(services.llm, ExtractiveLLM)
    # default embedding cache is "memory", so the offline embedder is wrapped:
    assert isinstance(services.embedder, CachingEmbedder)
    assert services.embedder.model_name == "hashing-384"
    assert isinstance(services.store, FaissVectorStore)

    await services.ingestion.ingest_text(
        "A honeybee colony has one queen and many female workers.", doc_id="bees", title="Bees"
    )
    answer = await services.query.answer(Query("queen and workers in a honeybee colony", top_k=2))
    assert answer.citations
    assert answer.sources[0].doc_id == "bees"
    await services.aclose()


async def test_openai_embedding_requires_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError):
        build_services(_settings(monkeypatch, tmp_path, RAG_EMBEDDING_PROVIDER="openai"))


async def test_openai_llm_requires_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        build_services(_settings(monkeypatch, tmp_path, RAG_LLM_PROVIDER="openai"))


async def test_real_providers_are_wrapped_for_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    services = build_services(
        _settings(
            monkeypatch,
            tmp_path,
            RAG_EMBEDDING_PROVIDER="openai",
            RAG_LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-test",
        )
    )
    assert isinstance(services.embedder, FallbackEmbedder)
    assert isinstance(services.llm, FallbackLLM)
    assert services.embedder.dimension == 1536  # text-embedding-3-small
    assert services.closeables  # the OpenAI httpx clients are tracked for shutdown
    await services.aclose()


async def test_fallback_disabled_uses_bare_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    services = build_services(
        _settings(
            monkeypatch,
            tmp_path,
            RAG_EMBEDDING_PROVIDER="openai",
            RAG_LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-test",
            RAG_FALLBACK_TO_OFFLINE="false",
        )
    )
    assert isinstance(services.embedder, CachingEmbedder)
    assert isinstance(services.llm, OpenAIChatLLM)
    assert not isinstance(services.llm, ExtractiveLLM)
    await services.aclose()
