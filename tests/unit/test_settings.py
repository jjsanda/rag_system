"""Unit tests for the pydantic-settings configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.config.settings import Settings


def test_reads_rag_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TOP_K", "9")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "openai")
    settings = Settings(_env_file=None)
    assert settings.top_k == 9
    assert settings.llm_provider == "openai"


def test_reads_unprefixed_secret_and_otel_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
    monkeypatch.setenv("OTEL_ENABLED", "true")
    settings = Settings(_env_file=None)
    assert settings.openai_api_key == "sk-abc"
    assert settings.otel_enabled is True


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("RAG_EMBEDDING_PROVIDER", "RAG_LLM_PROVIDER", "RAG_VECTOR_STORE"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.embedding_provider == "fake"
    assert settings.llm_provider == "extractive"
    assert settings.vector_store == "faiss"
    assert settings.api_key is None


def test_invalid_literal_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_VECTOR_STORE", "mongodb")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_cors_origin_list_parsing() -> None:
    assert Settings(_env_file=None, cors_origins="a, b ,c").cors_origin_list == ["a", "b", "c"]


def test_public_info_excludes_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    info = Settings(_env_file=None).public_info()
    assert info["embedding_provider"] == "fake"
    assert "sk-secret" not in str(info)
    assert {"embedding_provider", "llm_provider", "vector_store", "top_k"} <= set(info)
