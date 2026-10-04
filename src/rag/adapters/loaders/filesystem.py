"""Filesystem document loader for ``.md`` / ``.txt`` corpora.

Loads a single file or recurses a directory tree, deriving each document's title from a
leading Markdown ``# H1`` (falling back to the file stem) and a stable id from the path
relative to the corpus root.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from rag.domain.exceptions import IngestionError
from rag.domain.models import Document

logger = logging.getLogger(__name__)

__all__ = ["FilesystemLoader"]


class FilesystemLoader:
    """Implements the :class:`~rag.domain.ports.DocumentLoader` port over the local filesystem."""

    def __init__(
        self, patterns: Sequence[str] = ("*.md", "*.txt"), encoding: str = "utf-8"
    ) -> None:
        self._patterns = tuple(patterns)
        self._encoding = encoding

    def load(self, source: str | Path) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise IngestionError(f"corpus path does not exist: {path}")

        if path.is_file():
            files = [path]
        else:
            files = sorted({f for pattern in self._patterns for f in path.rglob(pattern)})

        documents: list[Document] = []
        for file in files:
            try:
                text = file.read_text(encoding=self._encoding)
            except (OSError, UnicodeDecodeError) as exc:
                raise IngestionError(f"could not read {file}: {exc}") from exc
            title, body = _split_title(text)
            documents.append(
                Document(
                    id=_doc_id(file, path),
                    text=body,
                    title=title or file.stem,
                    source=str(file),
                )
            )
        logger.info("loaded %d document(s) from %s", len(documents), path)
        return documents


def _doc_id(file: Path, root: Path) -> str:
    if file == root:
        return file.stem
    base = root if root.is_dir() else root.parent
    return file.relative_to(base).with_suffix("").as_posix()


def _split_title(text: str) -> tuple[str | None, str]:
    """Return ``(title, body)``, lifting a leading Markdown ``# H1`` out of the body.

    Keeping the title out of the chunk text avoids the answer echoing the document's
    heading (and rendering it as a heading in the UI); the title is preserved separately.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            body = "\n".join(lines[index + 1 :]).lstrip("\n")
            return stripped[2:].strip(), body
        return None, text
    return None, text
