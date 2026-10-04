"""Shared pytest fixtures for the domain unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.domain.chunking import TokenAwareChunker
from rag.domain.cleaning import DefaultTextCleaner
from rag.domain.pipeline import IngestionService, QueryService
from rag.domain.prompt import build_system_prompt
from tests.fakes import CharTokenizer, FakeEmbedder, FakeLLM, InMemoryStore


@pytest.fixture
def char_tokenizer() -> CharTokenizer:
    return CharTokenizer()


@pytest.fixture
def cleaner() -> DefaultTextCleaner:
    return DefaultTextCleaner()


@pytest.fixture
def stack() -> SimpleNamespace:
    """A fully wired offline pipeline (shared embedder + store) for pipeline tests."""
    embedder = FakeEmbedder()
    store = InMemoryStore()
    llm = FakeLLM()
    chunker = TokenAwareChunker(CharTokenizer(), chunk_size=512, chunk_overlap=64)
    ingest = IngestionService(
        cleaner=DefaultTextCleaner(), chunker=chunker, embedder=embedder, store=store
    )
    query = QueryService(
        embedder=embedder,
        store=store,
        llm=llm,
        system_prompt=build_system_prompt(),
        default_top_k=4,
    )
    return SimpleNamespace(ingest=ingest, query=query, llm=llm, store=store, embedder=embedder)
