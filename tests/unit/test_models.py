"""Unit tests for the immutable domain models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rag.domain.models import Chunk, Document, Timings


def test_document_defaults() -> None:
    doc = Document(id="d1", text="hello")
    assert doc.title is None
    assert doc.source is None
    assert doc.metadata == {}


def test_document_is_immutable() -> None:
    doc = Document(id="d1", text="hello")
    field_name = "text"  # set via a variable so mypy allows the frozen-field write
    with pytest.raises(FrozenInstanceError):
        setattr(doc, field_name, "changed")


def test_chunk_carries_provenance() -> None:
    chunk = Chunk(id="d1::0", doc_id="d1", index=0, text="hello", token_count=1, title="T")
    assert chunk.id == "d1::0"
    assert chunk.title == "T"


def test_timings_optional_embedding() -> None:
    timings = Timings(retrieval_ms=1.0, generation_ms=2.0, total_ms=3.0)
    assert timings.embedding_ms is None
