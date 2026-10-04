"""Tests for the console-script CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.cli import main
from rag.config import get_settings

_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"


def test_cli_help_and_info() -> None:
    assert main([]) == 0  # no subcommand prints help
    assert main(["info"]) == 0  # info needs no services / no I/O


def test_cli_ingest_then_query(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in ("RAG_EMBEDDING_PROVIDER", "RAG_LLM_PROVIDER", "RAG_VECTOR_STORE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAG_FAISS_PATH", str(tmp_path / "faiss"))
    monkeypatch.setenv("RAG_CORPUS_DIR", str(_CORPUS))
    get_settings.cache_clear()
    try:
        assert main(["ingest", "--sample"]) == 0
        assert main(["query", "How many castes are there in a honeybee colony?"]) == 0
    finally:
        get_settings.cache_clear()
