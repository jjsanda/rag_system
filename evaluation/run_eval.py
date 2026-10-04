"""Evaluation harness for the RAG pipeline.

Runs a small labeled QA set (``qa_dataset.json``) over the sample corpus using the fully
offline stack and computes the following (all math in pure Python, so it is auditable):

* **hit@k**: fraction of questions whose gold document appears in the top-k retrieved.
* **MRR**: mean reciprocal rank of the first gold document (1/rank, 0 if absent).
* **coverage**: fraction of each question's labelled key facts present in the answer text.
* **groundedness**: for each answer sentence, the best lexical recall against any
  retrieved chunk (share of the sentence's content words found in the context), averaged.
  A proxy for whether the answer is supported by the retrieved context: ~1.0 for an
  extractive answerer (verbatim from context), lower for paraphrase or hallucination.

Run with:  make eval   (or  python -m evaluation.run_eval --top-k 5)
Writes evaluation/results.json and evaluation/results.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

from rag.config.settings import Settings
from rag.domain.models import Query
from rag.factory import Services, build_services

_HERE = Path(__file__).resolve().parent
_DATASET = _HERE / "qa_dataset.json"
_CORPUS = _HERE.parent / "data" / "corpus"
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class QAItem:
    id: str
    question: str
    relevant_doc_ids: list[str]
    key_facts: list[str]


@dataclass(frozen=True)
class QAResult:
    id: str
    hit: bool
    reciprocal_rank: float
    coverage: float
    groundedness: float
    top_doc: str


def load_dataset() -> list[QAItem]:
    payload = json.loads(_DATASET.read_text(encoding="utf-8"))
    return [QAItem(**item) for item in payload["items"]]


def _terms(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _groundedness(answer_text: str, context_texts: list[str]) -> float:
    """Mean over answer sentences of the best lexical recall against any retrieved chunk."""
    context_term_sets = [_terms(text) for text in context_texts]
    scores: list[float] = []
    for sentence in _SENTENCE_RE.split(answer_text):
        sentence_terms = _terms(sentence)
        if not sentence_terms or not context_term_sets:
            continue
        scores.append(
            max(len(sentence_terms & ctx) / len(sentence_terms) for ctx in context_term_sets)
        )
    return mean(scores) if scores else 0.0


async def evaluate(services: Services, items: list[QAItem], top_k: int) -> list[QAResult]:
    results: list[QAResult] = []
    for item in items:
        # Explicit retrieval gives the ranked docs (hit@k/MRR) and full chunk texts (grounding).
        query_vector = (await services.embedder.embed([item.question]))[0]
        retrieved = await services.store.search(query_vector, top_k)
        retrieved_docs = [rc.chunk.doc_id for rc in retrieved]
        gold = set(item.relevant_doc_ids)
        rank = next((i + 1 for i, doc in enumerate(retrieved_docs) if doc in gold), 0)

        answer = await services.query.answer(Query(item.question, top_k=top_k))
        text_lower = answer.text.lower()
        covered = sum(1 for fact in item.key_facts if fact.lower() in text_lower)
        coverage = covered / len(item.key_facts) if item.key_facts else 0.0

        results.append(
            QAResult(
                id=item.id,
                hit=rank > 0,
                reciprocal_rank=1.0 / rank if rank else 0.0,
                coverage=coverage,
                groundedness=_groundedness(answer.text, [rc.chunk.text for rc in retrieved]),
                top_doc=retrieved_docs[0] if retrieved_docs else "-",
            )
        )
    return results


def summarize(results: list[QAResult], top_k: int) -> dict[str, float | int]:
    return {
        "questions": len(results),
        "top_k": top_k,
        f"hit@{top_k}": round(mean(r.hit for r in results), 3),
        "mrr": round(mean(r.reciprocal_rank for r in results), 3),
        "key_fact_coverage": round(mean(r.coverage for r in results), 3),
        "groundedness": round(mean(r.groundedness for r in results), 3),
    }


def render(console: Console, results: list[QAResult], summary: dict[str, float | int]) -> None:
    table = Table(title="RAG evaluation: per question", show_lines=False)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("hit", justify="center")
    table.add_column("RR", justify="right")
    table.add_column("coverage", justify="right")
    table.add_column("grounded", justify="right")
    table.add_column("top doc", style="dim")
    for r in results:
        table.add_row(
            r.id,
            "✅" if r.hit else "❌",
            f"{r.reciprocal_rank:.2f}",
            f"{r.coverage:.2f}",
            f"{r.groundedness:.2f}",
            r.top_doc,
        )
    console.print(table)

    summary_table = Table(title="Summary", show_header=False)
    for key, value in summary.items():
        summary_table.add_row(str(key), str(value))
    console.print(summary_table)


def write_reports(results: list[QAResult], summary: dict[str, float | int]) -> None:
    (_HERE / "results.json").write_text(
        json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Evaluation results",
        "",
        "_Generated by `make eval` over the offline stack (deterministic)._",
        "",
        "| metric | value |",
        "| --- | --- |",
        *[f"| {key} | {value} |" for key, value in summary.items()],
        "",
        "| question | hit | RR | coverage | groundedness | top doc |",
        "| --- | :-: | --: | --: | --: | --- |",
        *[
            f"| {r.id} | {'✅' if r.hit else '❌'} | {r.reciprocal_rank:.2f} | "
            f"{r.coverage:.2f} | {r.groundedness:.2f} | {r.top_doc} |"
            for r in results
        ],
        "",
    ]
    (_HERE / "results.md").write_text("\n".join(lines), encoding="utf-8")


async def _run(top_k: int) -> dict[str, float | int]:
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            _env_file=None,
            embedding_provider="fake",
            llm_provider="extractive",
            vector_store="faiss",
            faiss_path=str(Path(tmp) / "faiss"),
            corpus_dir=str(_CORPUS),
        )
        services = build_services(settings)
        try:
            await services.ingestion.ingest_source(settings.corpus_dir)
            results = await evaluate(services, load_dataset(), top_k)
        finally:
            await services.aclose()

    summary = summarize(results, top_k)
    console = Console()
    render(console, results, summary)
    write_reports(results, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline over the sample corpus."
    )
    parser.add_argument("--top-k", type=int, default=5, help="retrieval depth for hit@k / MRR")
    args = parser.parse_args()
    asyncio.run(_run(args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
