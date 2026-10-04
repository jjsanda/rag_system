"""Unit tests for the ingestion and query orchestration services."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.domain.chunking import TokenAwareChunker
from rag.domain.cleaning import DefaultTextCleaner
from rag.domain.exceptions import EmbeddingProviderError, IngestionError
from rag.domain.models import Document, Query
from rag.domain.pipeline import IngestionService, QueryService
from rag.domain.ports import DocumentLoader, Embedder, VectorStore
from rag.domain.prompt import build_system_prompt
from tests.fakes import (
    CharTokenizer,
    FailingEmbedder,
    FakeEmbedder,
    FakeLLM,
    InMemoryStore,
    MemoryResponseCache,
    StaticLoader,
    StreamFailingLLM,
)


def _ingestion_service(
    embedder: Embedder, store: VectorStore, loader: DocumentLoader | None = None
) -> IngestionService:
    return IngestionService(
        cleaner=DefaultTextCleaner(),
        chunker=TokenAwareChunker(CharTokenizer(), chunk_size=512, chunk_overlap=0),
        embedder=embedder,
        store=store,
        loader=loader,
    )


_DOCS = [
    ("alpha", "Photosynthesis converts light energy into chemical energy in chloroplasts."),
    ("beta", "A honeybee colony has one queen and many female workers."),
    ("gamma", "TCP is a connection oriented transport layer protocol."),
]


async def _ingest_corpus(stack: SimpleNamespace) -> None:
    for doc_id, text in _DOCS:
        await stack.ingest.ingest_text(text, doc_id=doc_id, title=doc_id.upper())


async def test_answer_is_grounded_and_cited(stack: SimpleNamespace) -> None:
    await _ingest_corpus(stack)
    answer = await stack.query.answer(Query("queen and workers in a honeybee colony", top_k=2))

    assert answer.text
    assert answer.citations == (1,)
    assert len(answer.sources) == 2
    assert answer.sources[0].doc_id == "beta"  # most lexically similar document ranks first
    assert answer.llm_provider == "fake"
    assert answer.timings.total_ms >= 0
    assert answer.timings.embedding_ms is not None


async def test_ingest_counts(stack: SimpleNamespace) -> None:
    result = await stack.ingest.ingest_text("Hello world.", doc_id="x", title="X")
    assert (result.documents, result.chunks, result.vectors) == (1, 1, 1)


async def test_blank_document_yields_no_chunks(stack: SimpleNamespace) -> None:
    result = await stack.ingest.ingest_text("   ", doc_id="x")
    assert result.chunks == 0
    assert result.vectors == 0


async def test_streaming_emits_ordered_events(stack: SimpleNamespace) -> None:
    await _ingest_corpus(stack)
    events = [e async for e in stack.query.stream(Query("honeybee colony queen", top_k=2))]
    kinds = [e.event for e in events]

    assert kinds[0] == "meta"
    assert kinds[1] == "sources"
    assert "token" in kinds
    assert kinds[-1] == "done"
    assert "timings" in events[-1].data
    assert "citations" in events[-1].data


async def test_response_cache_short_circuits_llm() -> None:
    embedder = FakeEmbedder()
    store = InMemoryStore()
    llm = FakeLLM()
    ingest = IngestionService(
        cleaner=DefaultTextCleaner(),
        chunker=TokenAwareChunker(CharTokenizer(), chunk_size=512, chunk_overlap=0),
        embedder=embedder,
        store=store,
    )
    await ingest.ingest_text("A honeybee colony has a queen.", doc_id="beta", title="BETA")

    query_svc = QueryService(
        embedder=embedder,
        store=store,
        llm=llm,
        system_prompt=build_system_prompt(),
        response_cache=MemoryResponseCache(),
    )
    query = Query("honeybee queen", top_k=1)
    first = await query_svc.answer(query)
    second = await query_svc.answer(query)

    assert llm.calls == 1  # second answer served from cache
    assert first.text == second.text


async def test_stream_emits_error_event_when_embedding_fails() -> None:
    query_svc = QueryService(
        embedder=FailingEmbedder(),
        store=InMemoryStore(),
        llm=FakeLLM(),
        system_prompt=build_system_prompt(),
    )
    events = [e async for e in query_svc.stream(Query("anything", top_k=1))]

    assert len(events) == 1
    assert events[0].event == "error"
    assert events[0].data["type"] == "EmbeddingProviderError"


async def test_ingest_propagates_provider_error() -> None:
    service = _ingestion_service(FailingEmbedder(), InMemoryStore())
    with pytest.raises(EmbeddingProviderError):
        await service.ingest_text("Hello world.", doc_id="x")


async def test_ingest_source_requires_a_loader() -> None:
    service = _ingestion_service(FakeEmbedder(), InMemoryStore())
    with pytest.raises(IngestionError):
        await service.ingest_source("anywhere")


async def test_ingest_source_uses_loader() -> None:
    docs = [Document(id="a", text="Hello world.", title="A")]
    service = _ingestion_service(FakeEmbedder(), InMemoryStore(), loader=StaticLoader(docs))
    result = await service.ingest_source("ignored")
    assert result.documents == 1
    assert result.chunks == 1


async def test_stream_error_during_generation() -> None:
    embedder = FakeEmbedder()
    store = InMemoryStore()
    await _ingestion_service(embedder, store).ingest_text(
        "A honeybee colony has a queen.", doc_id="beta", title="BETA"
    )
    query_svc = QueryService(
        embedder=embedder,
        store=store,
        llm=StreamFailingLLM(),
        system_prompt=build_system_prompt(),
    )
    events = [e async for e in query_svc.stream(Query("honeybee", top_k=1))]
    kinds = [e.event for e in events]

    assert kinds[0] == "meta"
    assert "token" in kinds  # one token is emitted before the failure
    assert kinds[-1] == "error"
    assert events[-1].data["type"] == "LLMProviderError"
