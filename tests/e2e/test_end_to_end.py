"""End-to-end test: ingest the real sample corpus through the offline stack and assert a
grounded, cited answer that resolves to the correct source document.

Uses the fully-offline default providers (hashing embedder + FAISS + extractive answerer),
so it runs deterministically with no network and no external services.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config.settings import Settings
from rag.domain.models import Query
from rag.factory import build_services

_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"


@pytest.mark.e2e
async def test_ingest_corpus_then_grounded_cited_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for key in ("RAG_EMBEDDING_PROVIDER", "RAG_LLM_PROVIDER", "RAG_VECTOR_STORE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAG_FAISS_PATH", str(tmp_path / "faiss"))
    monkeypatch.setenv("RAG_CORPUS_DIR", str(_CORPUS))

    services = build_services(Settings(_env_file=None))
    try:
        ingestion = await services.ingestion.ingest_source(services.settings.corpus_dir)
        assert ingestion.documents >= 8
        assert ingestion.chunks > 0
        assert ingestion.vectors == ingestion.chunks

        answer = await services.query.answer(
            Query("How many castes are there in a honeybee colony?", top_k=4)
        )

        # Grounded: the honeybee document was retrieved and is among the cited sources.
        assert answer.text
        assert answer.citations, "answer must cite at least one source"
        retrieved_docs = {source.doc_id for source in answer.sources}
        assert "honeybee-colonies" in retrieved_docs
        # Every citation marker resolves to a returned source...
        assert all(1 <= number <= len(answer.sources) for number in answer.citations)
        # ...and at least one citation resolves specifically to the honeybee document.
        cited_docs = {answer.sources[number - 1].doc_id for number in answer.citations}
        assert "honeybee-colonies" in cited_docs
        # The cited text is grounded in the corpus (a [n] marker appears in the answer).
        assert "[" in answer.text and "]" in answer.text
    finally:
        await services.aclose()
