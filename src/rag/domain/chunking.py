"""Token-aware, sentence-aligned chunking with overlap.

The chunker packs sentences into chunks bounded by a token budget, carries a configurable
token overlap between consecutive chunks, and hard-splits any single sentence that exceeds
the budget on its own. Token counting is delegated to an injected
:class:`~rag.domain.ports.Tokenizer`, so chunking is independent of any embedding model.

Example:
    >>> from rag.domain.models import Document
    >>> class WordTokenizer:  # a trivial whitespace tokenizer for the doctest
    ...     def encode(self, text): return list(range(len(text.split())))
    ...     def decode(self, tokens): return " ".join(map(str, tokens))
    ...     def count(self, text): return len(text.split())
    >>> chunker = TokenAwareChunker(WordTokenizer(), chunk_size=6, chunk_overlap=2)
    >>> doc = Document(id="d1", text="One two three. Four five six. Seven eight nine.")
    >>> chunks = chunker.chunk_document(doc)
    >>> [c.token_count for c in chunks]
    [6, 3]
    >>> all(c.token_count <= 6 for c in chunks)
    True
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag.domain.exceptions import ConfigError
from rag.domain.models import Chunk, Document
from rag.domain.ports import Tokenizer

__all__ = ["TokenAwareChunker"]

# Split a paragraph into sentences: after sentence-ending punctuation followed by
# whitespace, or on any newline (keeps Markdown headings/list items as their own unit).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_PARAGRAPH_SPLIT = re.compile(r"\n{2,}")


class TokenAwareChunker:
    """Implements the chunking half of the ingestion pipeline.

    Args:
        tokenizer: token codec used to measure and split text.
        chunk_size: maximum number of tokens per chunk.
        chunk_overlap: number of trailing tokens carried into the next chunk
            (must be smaller than ``chunk_size``).
    """

    def __init__(
        self, tokenizer: Tokenizer, chunk_size: int = 400, chunk_overlap: int = 80
    ) -> None:
        if chunk_size < 1:
            raise ConfigError(f"chunk_size must be >= 1, got {chunk_size}")
        if not 0 <= chunk_overlap < chunk_size:
            raise ConfigError(
                f"chunk_overlap must satisfy 0 <= overlap < chunk_size; "
                f"got overlap={chunk_overlap}, chunk_size={chunk_size}"
            )
        self._tok = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # -- public API ---------------------------------------------------------------------

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split a single document into ordered, token-bounded chunks."""
        packed = self._pack(self._segments(doc.text))
        return [
            Chunk(
                id=f"{doc.id}::{index}",
                doc_id=doc.id,
                index=index,
                text=text,
                token_count=tokens,
                title=doc.title,
                source=doc.source,
                metadata=doc.metadata,
            )
            for index, (text, tokens) in enumerate(packed)
        ]

    def chunk_documents(self, docs: Sequence[Document]) -> list[Chunk]:
        """Chunk many documents, preserving order."""
        chunks: list[Chunk] = []
        for doc in docs:
            chunks.extend(self.chunk_document(doc))
        return chunks

    # -- internals ----------------------------------------------------------------------

    def _segments(self, text: str) -> list[str]:
        """Break text into trimmed, non-empty sentence-level segments."""
        segments: list[str] = []
        for paragraph in _PARAGRAPH_SPLIT.split(text):
            for sentence in _SENTENCE_SPLIT.split(paragraph):
                stripped = sentence.strip()
                if stripped:
                    segments.append(stripped)
        return segments

    @staticmethod
    def _join(segments: Sequence[str]) -> str:
        return " ".join(segments)

    def _pack(self, segments: Sequence[str]) -> list[tuple[str, int]]:
        """Greedily pack segments into ``(text, token_count)`` chunks within the budget."""
        chunks: list[tuple[str, int]] = []
        current: list[str] = []

        for segment in segments:
            seg_tokens = self._tok.count(segment)

            # A single oversized sentence is flushed-around and hard-split by tokens.
            if seg_tokens > self.chunk_size:
                self._flush(current, chunks)
                current = []
                chunks.extend(self._hard_split(segment))
                continue

            # Fits in the current chunk? Measure the joined text for an exact token bound.
            if current and self._tok.count(self._join([*current, segment])) <= self.chunk_size:
                current.append(segment)
                continue
            if not current:
                current = [segment]
                continue

            # Otherwise finalise the chunk and seed the next one with token overlap.
            self._flush(current, chunks)
            budget = min(self.chunk_overlap, self.chunk_size - seg_tokens)
            seed = [*self._trailing(current, budget), segment]
            if self._tok.count(self._join(seed)) > self.chunk_size:
                seed = [segment]  # overlap + segment would overflow; drop the overlap
            current = seed

        self._flush(current, chunks)
        return chunks

    def _flush(self, segments: Sequence[str], chunks: list[tuple[str, int]]) -> None:
        if not segments:
            return
        text = self._join(segments)
        chunks.append((text, self._tok.count(text)))

    def _trailing(self, segments: Sequence[str], budget: int) -> list[str]:
        """Return the longest suffix of ``segments`` whose tokens fit within ``budget``."""
        if budget <= 0:
            return []
        chosen: list[str] = []
        total = 0
        for segment in reversed(segments):
            tokens = self._tok.count(segment)
            if total + tokens > budget:
                break
            chosen.append(segment)
            total += tokens
        chosen.reverse()
        return chosen

    def _hard_split(self, segment: str) -> list[tuple[str, int]]:
        """Split one oversized sentence into overlapping token windows."""
        tokens = self._tok.encode(segment)
        stride = self.chunk_size - self.chunk_overlap
        windows: list[tuple[str, int]] = []
        start = 0
        while start < len(tokens):
            window = tokens[start : start + self.chunk_size]
            windows.append((self._tok.decode(window), len(window)))
            if start + self.chunk_size >= len(tokens):
                break
            start += stride
        return windows
