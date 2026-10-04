"""The two core orchestration services: ingestion and query.

They depend only on domain models, ports and the
chunker/prompt helpers, never on a framework or a concrete provider. Timing is captured
here (retrieval vs. generation separately) and returned in the result so the observability
layer can record it without the domain importing Prometheus.

Example (fully offline, see tests for the runnable version):
    >>> import asyncio
    >>> from rag.domain.models import Document, Query
    >>> async def demo(ingest, query_svc):
    ...     await ingest.ingest_text("Bees build hexagonal comb.", doc_id="bees", title="Bees")
    ...     answer = await query_svc.answer(Query("What shape is honeycomb?", top_k=1))
    ...     return answer.text, answer.citations
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from time import perf_counter

from rag.domain.chunking import TokenAwareChunker
from rag.domain.exceptions import IngestionError, RagError
from rag.domain.models import (
    Answer,
    Document,
    EmbeddedChunk,
    IngestionResult,
    Query,
    Source,
    StreamEvent,
    Timings,
)
from rag.domain.ports import LLM, DocumentLoader, Embedder, ResponseCache, TextCleaner, VectorStore
from rag.domain.prompt import assemble_prompt, extract_citation_numbers

__all__ = ["IngestionService", "QueryService"]

_WHITESPACE_RE = re.compile(r"\s+")


def _flatten(title: str | None) -> str | None:
    """Collapse a document title to a single clean line (titles are untrusted input)."""
    if not title:
        return None
    return _WHITESPACE_RE.sub(" ", title).strip() or None


def _ms_since(start: float) -> float:
    return (perf_counter() - start) * 1000.0


class IngestionService:
    """Loads, cleans, chunks, embeds and stores documents."""

    def __init__(
        self,
        *,
        cleaner: TextCleaner,
        chunker: TokenAwareChunker,
        embedder: Embedder,
        store: VectorStore,
        loader: DocumentLoader | None = None,
    ) -> None:
        self._cleaner = cleaner
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._loader = loader

    async def ingest_documents(self, docs: Sequence[Document]) -> IngestionResult:
        """Run the full ingestion pipeline for a batch of documents."""
        start = perf_counter()
        try:
            cleaned = [
                Document(
                    id=doc.id,
                    text=self._cleaner.clean(doc.text),
                    title=_flatten(doc.title),
                    source=doc.source,
                    metadata=doc.metadata,
                )
                for doc in docs
            ]
            chunks = self._chunker.chunk_documents(cleaned)
            if not chunks:
                return IngestionResult(len(docs), 0, 0, _ms_since(start))

            vectors = await self._embedder.embed([chunk.text for chunk in chunks])
            embedded = [
                EmbeddedChunk(chunk=chunk, embedding=vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            await self._store.upsert(embedded)
            return IngestionResult(len(docs), len(chunks), len(embedded), _ms_since(start))
        except RagError:
            raise
        except Exception as exc:  # pragma: no cover
            raise IngestionError(f"ingestion failed: {exc}") from exc

    async def ingest_text(
        self,
        text: str,
        *,
        doc_id: str,
        title: str | None = None,
        source: str | None = None,
    ) -> IngestionResult:
        """Ingest a single in-memory document."""
        document = Document(id=doc_id, text=text, title=title, source=source)
        return await self.ingest_documents([document])

    async def ingest_source(self, source: str | Path) -> IngestionResult:
        """Load documents from a configured loader and ingest them."""
        if self._loader is None:
            raise IngestionError("no document loader is configured")
        return await self.ingest_documents(self._loader.load(source))


class QueryService:
    """Embeds a query, retrieves context, assembles a grounded prompt and calls the LLM."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        system_prompt: str,
        default_top_k: int = 4,
        snippet_chars: int = 240,
        response_cache: ResponseCache | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._system_prompt = system_prompt
        self._default_top_k = default_top_k
        self._snippet_chars = snippet_chars
        self._cache = response_cache

    # -- non-streaming --------------------------------------------------------------------

    async def answer(self, query: Query) -> Answer:
        """Produce a complete grounded answer with citations and a latency breakdown."""
        cache_key = self._cache_key(query)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        prompt, sources, retrieval_ms, embedding_ms = await self._retrieve(query)

        gen_start = perf_counter()
        text = await self._llm.generate(system=self._system_prompt, prompt=prompt)
        generation_ms = _ms_since(gen_start)

        answer = Answer(
            text=text,
            sources=tuple(sources),
            citations=extract_citation_numbers(text, len(sources)),
            timings=Timings(
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                total_ms=retrieval_ms + generation_ms,
                embedding_ms=embedding_ms,
            ),
            model=self._llm.model_name,
            llm_provider=self._llm.provider,
        )
        if self._cache is not None:
            self._cache.set(cache_key, answer)
        return answer

    # -- streaming ------------------------------------------------------------------------

    async def stream(self, query: Query) -> AsyncIterator[StreamEvent]:
        """Yield SSE events: ``meta`` → ``sources`` → ``token``* → ``done`` (or ``error``).

        Retrieval failures before the first token surface as an ``error`` event so the SSE
        response always terminates cleanly (the HTTP status is already ``200`` once
        streaming begins).
        """
        try:
            prompt, sources, retrieval_ms, embedding_ms = await self._retrieve(query)
        except RagError as exc:
            yield StreamEvent("error", {"message": str(exc), "type": type(exc).__name__})
            return

        yield StreamEvent(
            "meta",
            {"model": self._llm.model_name, "provider": self._llm.provider, "top_k": query.top_k},
        )
        yield StreamEvent("sources", {"sources": [_source_dict(s) for s in sources]})

        parts: list[str] = []
        gen_start = perf_counter()
        try:
            async for delta in self._llm.stream(system=self._system_prompt, prompt=prompt):
                parts.append(delta)
                yield StreamEvent("token", {"text": delta})
        except RagError as exc:
            yield StreamEvent("error", {"message": str(exc), "type": type(exc).__name__})
            return

        generation_ms = _ms_since(gen_start)
        full = "".join(parts)
        yield StreamEvent(
            "done",
            {
                "citations": list(extract_citation_numbers(full, len(sources))),
                "timings": {
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": round(generation_ms, 2),
                    "total_ms": round(retrieval_ms + generation_ms, 2),
                    "embedding_ms": round(embedding_ms, 2),
                },
            },
        )

    # -- internals ------------------------------------------------------------------------

    async def _retrieve(self, query: Query) -> tuple[str, list[Source], float, float]:
        """Embed the query and search the store; return prompt, sources and timings.

        ``retrieval_ms`` includes the query-embedding time (reported separately as
        ``embedding_ms``) plus the vector search.
        """
        top_k = max(1, query.top_k or self._default_top_k)

        embed_start = perf_counter()
        query_vector = (await self._embedder.embed([query.text]))[0]
        embedding_ms = _ms_since(embed_start)

        search_start = perf_counter()
        retrieved = await self._store.search(query_vector, top_k)
        retrieval_ms = embedding_ms + _ms_since(search_start)

        prompt, sources = assemble_prompt(query.text, retrieved, snippet_chars=self._snippet_chars)
        return prompt, sources, retrieval_ms, embedding_ms

    def _cache_key(self, query: Query) -> str:
        raw = "\x1f".join(
            [
                query.text,
                str(query.top_k),
                self._embedder.model_name,
                self._llm.provider,
                self._llm.model_name,
                self._system_prompt,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _source_dict(source: Source) -> dict[str, object]:
    return {
        "number": source.number,
        "doc_id": source.doc_id,
        "chunk_id": source.chunk_id,
        "title": source.title,
        "score": round(source.score, 4),
        "snippet": source.snippet,
    }
