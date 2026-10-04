"""Unit tests for the token-aware chunker (using a char-level tokenizer)."""

from __future__ import annotations

import pytest

from rag.domain.chunking import TokenAwareChunker
from rag.domain.exceptions import ConfigError
from rag.domain.models import Document
from tests.fakes import CharTokenizer


def _chunker(size: int = 40, overlap: int = 10) -> TokenAwareChunker:
    return TokenAwareChunker(CharTokenizer(), chunk_size=size, chunk_overlap=overlap)


def test_rejects_invalid_parameters() -> None:
    with pytest.raises(ConfigError):
        TokenAwareChunker(CharTokenizer(), chunk_size=0, chunk_overlap=0)
    with pytest.raises(ConfigError):
        TokenAwareChunker(CharTokenizer(), chunk_size=10, chunk_overlap=10)
    with pytest.raises(ConfigError):
        TokenAwareChunker(CharTokenizer(), chunk_size=10, chunk_overlap=20)


def test_empty_document_yields_no_chunks() -> None:
    assert _chunker().chunk_document(Document(id="d", text="   \n\n  ")) == []


def test_respects_token_budget_and_indexing() -> None:
    text = " ".join(f"Sentence number {i} about a topic." for i in range(20))
    chunks = _chunker(size=80, overlap=16).chunk_document(Document(id="d", text=text, title="T"))

    assert len(chunks) >= 2
    assert all(c.token_count <= 80 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.id == f"d::{c.index}" for c in chunks)
    assert all(c.title == "T" for c in chunks)


def test_hard_split_of_oversized_segment_has_exact_overlap() -> None:
    # A single 100-char "word" (no spaces/punctuation) forces token-window splitting.
    text = "".join(chr(ord("a") + (i % 26)) for i in range(100))
    chunks = _chunker(size=40, overlap=10).chunk_document(Document(id="d", text=text))

    assert len(chunks) == 3
    assert all(c.token_count <= 40 for c in chunks)
    # The 10-token overlap means chunk N's tail equals chunk N+1's head.
    assert chunks[0].text[-10:] == chunks[1].text[:10]
    assert chunks[1].text[-10:] == chunks[2].text[:10]


def test_chunk_documents_flattens_and_preserves_doc_ids() -> None:
    docs = [Document(id="a", text="Hello world."), Document(id="b", text="Foo bar baz.")]
    chunks = _chunker(size=512, overlap=0).chunk_documents(docs)
    assert {c.doc_id for c in chunks} == {"a", "b"}
