"""Unit tests for the evaluation harness metrics."""

from __future__ import annotations

from pathlib import Path

from evaluation.run_eval import QAResult, _groundedness, load_dataset, summarize

_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"


def test_groundedness_full_for_verbatim_sentence() -> None:
    context = ["Honeybees build hexagonal comb from beeswax in the hive."]
    assert _groundedness("Honeybees build hexagonal comb.", context) == 1.0


def test_groundedness_zero_for_unrelated_sentence() -> None:
    assert _groundedness("Quantum chromodynamics governs gluons.", ["The cat sat down."]) == 0.0


def test_groundedness_handles_empty() -> None:
    assert _groundedness("", ["anything"]) == 0.0
    assert _groundedness("Some text.", []) == 0.0


def test_dataset_references_existing_documents() -> None:
    stems = {p.stem for p in _CORPUS.glob("*.md")}
    items = load_dataset()
    assert len(items) >= 10
    for item in items:
        assert item.question and item.key_facts
        assert all(doc_id in stems for doc_id in item.relevant_doc_ids)


def test_summarize_aggregates() -> None:
    results = [
        QAResult(
            id="a", hit=True, reciprocal_rank=1.0, coverage=1.0, groundedness=1.0, top_doc="x"
        ),
        QAResult(
            id="b", hit=False, reciprocal_rank=0.0, coverage=0.0, groundedness=0.5, top_doc="y"
        ),
    ]
    summary = summarize(results, top_k=5)
    assert summary["questions"] == 2
    assert summary["hit@5"] == 0.5
    assert summary["mrr"] == 0.5
    assert summary["groundedness"] == 0.75
