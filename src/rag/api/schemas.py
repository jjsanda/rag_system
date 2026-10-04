"""Pydantic request/response models: the API's typed boundary (OpenAPI source of truth)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag.domain.models import Answer, IngestionResult

__all__ = [
    "QueryRequest",
    "IngestDocument",
    "IngestRequest",
    "SourceModel",
    "TimingsModel",
    "QueryResponse",
    "IngestResponse",
    "InfoResponse",
    "HealthResponse",
    "ReadyResponse",
    "ErrorBody",
    "ErrorResponse",
]


# Hard backstops rejected during deserialization, before the configurable RAG_MAX_* checks
# (which the routes still apply). They bound the request body so a pathological payload is
# refused early.
_MAX_QUESTION_LEN = 16_000
_MAX_DOCUMENT_LEN = 1_000_000
_MAX_DOCUMENTS = 1_000


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=_MAX_QUESTION_LEN, description="The question.")
    top_k: int | None = Field(default=None, ge=1, le=50, description="Override retrieval depth.")


class IngestDocument(BaseModel):
    text: str = Field(min_length=1, max_length=_MAX_DOCUMENT_LEN)
    id: str | None = None
    title: str | None = Field(default=None, max_length=2_000)
    source: str | None = None


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(default_factory=list, max_length=_MAX_DOCUMENTS)
    use_sample_corpus: bool = Field(
        default=False, description="Ingest the bundled sample corpus instead of `documents`."
    )


class SourceModel(BaseModel):
    number: int
    doc_id: str
    chunk_id: str
    title: str | None
    score: float
    snippet: str


class TimingsModel(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    embedding_ms: float | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[int]
    sources: list[SourceModel]
    timings: TimingsModel
    model: str
    llm_provider: str
    request_id: str | None = None

    @classmethod
    def from_answer(cls, answer: Answer, request_id: str | None) -> QueryResponse:
        return cls(
            answer=answer.text,
            citations=list(answer.citations),
            sources=[
                SourceModel(
                    number=s.number,
                    doc_id=s.doc_id,
                    chunk_id=s.chunk_id,
                    title=s.title,
                    score=s.score,
                    snippet=s.snippet,
                )
                for s in answer.sources
            ],
            timings=TimingsModel(
                retrieval_ms=answer.timings.retrieval_ms,
                generation_ms=answer.timings.generation_ms,
                total_ms=answer.timings.total_ms,
                embedding_ms=answer.timings.embedding_ms,
            ),
            model=answer.model,
            llm_provider=answer.llm_provider,
            request_id=request_id,
        )


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    vectors: int
    took_ms: float

    @classmethod
    def from_result(cls, result: IngestionResult) -> IngestResponse:
        return cls(
            documents=result.documents,
            chunks=result.chunks,
            vectors=result.vectors,
            took_ms=round(result.took_ms, 2),
        )


class InfoResponse(BaseModel):
    version: str
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    vector_store: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    fallback_to_offline: bool


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: str
    degraded: bool
    checks: dict[str, bool]


class ErrorBody(BaseModel):
    type: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str | None = None
