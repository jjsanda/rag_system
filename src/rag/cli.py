"""Command-line interface for the RAG system.

A thin wrapper over the same wired services the API uses, for quick local use:

    rag info                       # show the effective configuration
    rag ingest --sample            # ingest the bundled sample corpus
    rag ingest path/to/docs        # ingest a file or directory
    rag query "your question"      # ask a grounded, cited question

Honours the same environment configuration as the service (see ``.env.example``).
"""

from __future__ import annotations

import argparse
import asyncio

from rag.config import get_settings
from rag.domain.models import Query
from rag.factory import build_services


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point referenced by ``[project.scripts]``."""
    parser = argparse.ArgumentParser(prog="rag", description="Retrieval-Augmented Generation CLI.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Show the effective (non-secret) configuration.")

    ingest = sub.add_parser("ingest", help="Ingest documents into the vector store.")
    ingest.add_argument("path", nargs="?", help="File or directory to ingest.")
    ingest.add_argument("--sample", action="store_true", help="Ingest the bundled sample corpus.")

    ask = sub.add_parser("query", help="Ask a grounded question.")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None, help="Number of passages to retrieve.")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()

    if args.command == "info":
        for key, value in settings.public_info().items():
            print(f"{key}: {value}")
        return 0

    services = build_services(settings)
    try:
        if args.command == "ingest":
            source = args.path if args.path else settings.corpus_dir
            result = await services.ingestion.ingest_source(source)
            print(
                f"ingested {result.documents} document(s) -> {result.chunks} chunks "
                f"({result.vectors} vectors, {result.took_ms:.0f} ms)"
            )
        elif args.command == "query":
            answer = await services.query.answer(
                Query(args.question, top_k=args.top_k or settings.top_k)
            )
            print(answer.text)
            print()
            for source in answer.sources:
                cited = "*" if source.number in answer.citations else " "
                print(
                    f" [{cited}{source.number}] {source.title or source.doc_id}  ({source.score:.3f})"
                )
    finally:
        await services.aclose()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
