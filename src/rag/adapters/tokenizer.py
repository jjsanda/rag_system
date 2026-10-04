"""Tokenizer adapter backed by OpenAI's ``tiktoken`` BPE encodings.

``tiktoken`` downloads the chosen encoding's rank file on first use and caches it on disk
(set ``TIKTOKEN_CACHE_DIR`` to control where). The Docker image pre-fetches it at build
time so the container is fully offline-capable.

Example:
    >>> tok = TiktokenTokenizer("cl100k_base")          # doctest: +SKIP
    >>> tok.decode(tok.encode("hello world")) == "hello world"
    True
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import tiktoken

from rag.domain.exceptions import ConfigError

logger = logging.getLogger(__name__)

__all__ = ["TiktokenTokenizer"]


class TiktokenTokenizer:
    """Implements the :class:`~rag.domain.ports.Tokenizer` port via ``tiktoken``."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        try:
            self._encoding = tiktoken.get_encoding(encoding)
        except Exception as exc:  # network failure / unknown name
            raise ConfigError(f"could not load tiktoken encoding {encoding!r}: {exc}") from exc
        self.encoding_name = encoding
        logger.debug("loaded tiktoken encoding %s", encoding)

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: Sequence[int]) -> str:
        return self._encoding.decode(list(tokens))

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))
