"""The framework-free RAG core: models, ports and orchestration.

Re-exports the most commonly used names so callers can write
``from rag.domain import Document, QueryService`` instead of reaching into submodules.
"""

from __future__ import annotations

from rag.domain.chunking import TokenAwareChunker
from rag.domain.cleaning import DefaultTextCleaner
from rag.domain.exceptions import (
    ConfigError,
    EmbeddingProviderError,
    IngestionError,
    InputValidationError,
    LLMProviderError,
    ProviderError,
    ProviderUnavailableError,
    RagError,
    RateLimitError,
    RetrievalError,
    VectorStoreError,
)
from rag.domain.models import (
    Answer,
    Chunk,
    Document,
    EmbeddedChunk,
    IngestionResult,
    Query,
    RetrievedChunk,
    Source,
    StreamEvent,
    Timings,
)
from rag.domain.pipeline import IngestionService, QueryService
from rag.domain.ports import (
    LLM,
    DocumentLoader,
    Embedder,
    EmbeddingCache,
    ResponseCache,
    TextCleaner,
    Tokenizer,
    VectorStore,
)

__all__ = [
    # models
    "Answer",
    "Chunk",
    "Document",
    "EmbeddedChunk",
    "IngestionResult",
    "Query",
    "RetrievedChunk",
    "Source",
    "StreamEvent",
    "Timings",
    # services + domain logic
    "IngestionService",
    "QueryService",
    "TokenAwareChunker",
    "DefaultTextCleaner",
    # ports
    "DocumentLoader",
    "Embedder",
    "EmbeddingCache",
    "LLM",
    "ResponseCache",
    "TextCleaner",
    "Tokenizer",
    "VectorStore",
    # exceptions
    "RagError",
    "ConfigError",
    "InputValidationError",
    "IngestionError",
    "RetrievalError",
    "ProviderError",
    "EmbeddingProviderError",
    "LLMProviderError",
    "VectorStoreError",
    "ProviderUnavailableError",
    "RateLimitError",
]
