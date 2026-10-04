"""Ingestion endpoint: ingest provided documents or the bundled sample corpus."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rag.api.deps import ServicesDep, verify_api_key
from rag.api.schemas import IngestRequest, IngestResponse
from rag.domain.exceptions import InputValidationError
from rag.domain.models import Document
from rag.observability.metrics import INGESTED_CHUNKS, INGESTED_DOCUMENTS

router = APIRouter(prefix="/api/v1", tags=["ingest"], dependencies=[Depends(verify_api_key)])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(payload: IngestRequest, services: ServicesDep) -> IngestResponse:
    settings = services.settings

    if payload.use_sample_corpus:
        result = await services.ingestion.ingest_source(settings.corpus_dir)
    elif payload.documents:
        documents = []
        for index, item in enumerate(payload.documents):
            if len(item.text) > settings.max_document_chars:
                raise InputValidationError(
                    f"document {index} exceeds the maximum of "
                    f"{settings.max_document_chars} characters"
                )
            documents.append(
                Document(
                    id=item.id or f"doc-{index}",
                    text=item.text,
                    title=item.title,
                    source=item.source,
                )
            )
        result = await services.ingestion.ingest_documents(documents)
    else:
        raise InputValidationError("provide `documents` or set `use_sample_corpus=true`")

    INGESTED_DOCUMENTS.inc(result.documents)
    INGESTED_CHUNKS.inc(result.chunks)
    return IngestResponse.from_result(result)
