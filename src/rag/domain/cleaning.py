"""Default text cleaning.

A conservative cleaner: it normalises encoding and whitespace and strips control
characters, but deliberately preserves Markdown structure (headings, lists) so that
downstream sentence/paragraph splitting still has boundaries to work with.

Example:
    >>> DefaultTextCleaner().clean("Hello\\x00   world\\n\\n\\n\\nbye  ")
    'Hello world\\n\\nbye'
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["DefaultTextCleaner"]

# Control characters except tab/newline/carriage-return.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Runs of spaces/tabs (not newlines).
_INLINE_WS = re.compile(r"[ \t]+")
# 3+ consecutive newlines collapse to a paragraph break.
_BLANK_LINES = re.compile(r"\n{3,}")
# Trailing spaces before a newline.
_TRAILING_WS = re.compile(r"[ \t]+\n")


class DefaultTextCleaner:
    """Framework-free implementation of the :class:`~rag.domain.ports.TextCleaner` port."""

    def clean(self, text: str) -> str:
        """Normalise unicode and whitespace and drop control characters."""
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _CONTROL_CHARS.sub("", text)
        text = _INLINE_WS.sub(" ", text)
        text = _TRAILING_WS.sub("\n", text)
        text = _BLANK_LINES.sub("\n\n", text)
        return text.strip()
