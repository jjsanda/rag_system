# 1. Hexagonal (ports & adapters) architecture

- **Status:** Accepted
- **Date:** 2026-06-29

## Context

A RAG system has many interchangeable moving parts (embedding models, LLMs, vector stores)
and two delivery surfaces (HTTP API, UI). Coupling the retrieval/generation logic to any
one framework or provider would make it hard to test, to swap providers, or to reason about.

## Decision

Organise the code as a hexagon. `rag.domain` holds the framework-free core: immutable
dataclasses, `typing.Protocol` ports, the token-aware chunker, prompt assembly and the
ingestion/query pipeline. Every external capability is a port; concrete adapters under
`rag.adapters` implement them. A single composition root (`rag.factory`) reads `Settings`
and wires adapters into the core. FastAPI and Streamlit are the driving adapters; providers
and stores are the driven ones.

## Consequences

- The core has zero imports of FastAPI/Streamlit/OpenAI/Qdrant, so it is unit-tested with
  tiny in-memory fakes and no network.
- Swapping a provider or store is a configuration change in one place, not a code change.
- Slightly more indirection (ports + a composition root) than a monolithic design, in
  exchange for testability and substitutability.
