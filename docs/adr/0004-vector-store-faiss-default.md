# 4. FAISS as the default vector store, Qdrant for production

- **Status:** Accepted
- **Date:** 2026-06-29

## Context

The local default should need no server, while production deployments want a real, scalable
vector database. The `VectorStore` port lets us offer both behind one interface.

## Decision

Default to an in-process FAISS flat index (`IndexFlatIP` over L2-normalised vectors, so
inner product = cosine), with optional JSON persistence. Provide Chroma (embedded, also
local) and Qdrant (client/server, the production option) as drop-in adapters selected by
`RAG_VECTOR_STORE`. Qdrant maps string chunk ids to deterministic UUID point ids and is run
as a Docker/Helm service (tests use its in-memory mode).

## Consequences

- `pip install` + run works with no database; the offline default needs no separate service.
- Production scale is a config switch to Qdrant, with no application code changes.
- FAISS persistence here is a simple JSON snapshot suited to demo-scale corpora; large
  corpora or multi-replica deployments should use Qdrant.

## Alternatives considered

Postgres with `pgvector` and a hosted vector database were also on the table. `pgvector` fits
well when a project already runs Postgres, but it adds a service the offline default is meant
to avoid; a hosted database conflicts with the no-keys default. FAISS by default, with Qdrant
as the production switch, covers both ends without forcing infrastructure on the simple case.
