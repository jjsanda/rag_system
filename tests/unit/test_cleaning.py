"""Unit tests for the default text cleaner."""

from __future__ import annotations

from rag.domain.cleaning import DefaultTextCleaner


def test_collapses_whitespace_and_strips_controls() -> None:
    cleaned = DefaultTextCleaner().clean("Hello\x00   world\n\n\n\nbye  ")
    assert cleaned == "Hello world\n\nbye"


def test_normalizes_crlf() -> None:
    assert DefaultTextCleaner().clean("a\r\nb\r\nc") == "a\nb\nc"


def test_unicode_nfc_normalization() -> None:
    # "e" + combining acute accent should normalise to the single code point "é".
    assert DefaultTextCleaner().clean("é") == "é"


def test_preserves_paragraph_breaks() -> None:
    cleaned = DefaultTextCleaner().clean("Para one.\n\nPara two.")
    assert cleaned == "Para one.\n\nPara two."
