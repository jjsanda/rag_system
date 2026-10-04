"""Deterministic extractive answerer: the offline default LLM.

Instead of calling a model, it reconstructs the same numbered context a real LLM would see
(via :func:`rag.domain.prompt.parse_context`), scores each context sentence by lexical
overlap with the question, and stitches the best few together, each tagged with its
``[n]`` source marker. The result is grounded, always cited, and perfectly reproducible, so
the project runs end-to-end (and its tests/eval pass) with zero external services.

Example:
    >>> import asyncio
    >>> from rag.domain.models import Chunk, RetrievedChunk
    >>> from rag.domain.prompt import assemble_prompt, build_system_prompt
    >>> rc = RetrievedChunk(
    ...     Chunk("d::0", "d", 0, "Honeybees build hexagonal comb. Drones are male bees.", 9,
    ...           title="Bees"),
    ...     0.9,
    ... )
    >>> prompt, _ = assemble_prompt("What shape is honeycomb?", [rc])
    >>> answer = asyncio.run(ExtractiveLLM().generate(system=build_system_prompt(), prompt=prompt))
    >>> "[1]" in answer
    True
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import NamedTuple

from rag.domain.prompt import parse_context, parse_question

__all__ = ["ExtractiveLLM"]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    ]
)
_NO_CONTEXT = "I do not have enough information in the provided context to answer that question."


class _Candidate(NamedTuple):
    sentence: str
    number: int
    order: int
    score: int


class ExtractiveLLM:
    """Implements the :class:`~rag.domain.ports.LLM` port without any external model."""

    def __init__(self, max_sentences: int = 3, model_name: str = "extractive-v1") -> None:
        self._max_sentences = max_sentences
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "extractive"

    async def generate(self, *, system: str, prompt: str) -> str:
        return self._answer(prompt)

    async def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]:
        for token in _stream_tokens(self._answer(prompt)):
            yield token

    def _answer(self, prompt: str) -> str:
        blocks = parse_context(prompt)
        if not blocks:
            return _NO_CONTEXT

        question_terms = _terms(parse_question(prompt))
        candidates: list[_Candidate] = []
        order = 0
        for block in blocks:
            for sentence in _sentences(block.text):
                score = len(question_terms & _terms(sentence))
                candidates.append(_Candidate(sentence, block.number, order, score))
                order += 1

        if not candidates:
            return _NO_CONTEXT

        ranked = sorted(candidates, key=lambda c: (-c.score, c.order))
        chosen = [c for c in ranked if c.score > 0][: self._max_sentences] or [ranked[0]]
        chosen.sort(key=lambda c: c.order)  # present in original reading order
        return " ".join(f"{c.sentence} [{c.number}]" for c in chosen)


def _terms(text: str) -> set[str]:
    return {word for word in _WORD_RE.findall(text.lower()) if word not in _STOPWORDS}


def _sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _stream_tokens(text: str) -> list[str]:
    """Split an answer into word-sized streaming deltas (trailing space preserved)."""
    return [f"{word} " for word in text.split(" ") if word] or [text]
