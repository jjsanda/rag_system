# 3. Offline-deterministic default stack

- **Status:** Accepted
- **Date:** 2026-06-29

## Context

The project must run end-to-end on a fresh clone with no API keys and no external services,
and CI must be green and fast without secrets. RAG demos that require an LLM key or a running
vector DB to do anything are painful to evaluate.

## Decision

Ship two deterministic, dependency-light providers as the defaults:

- `HashingEmbedder` (`fake`): signed feature hashing into a fixed-dim unit vector. No model,
  no network; cosine similarity still tracks lexical overlap, so retrieval is meaningful.
- `ExtractiveLLM` (`extractive`): builds a cited answer from the same numbered prompt a real
  LLM sees, by selecting the most relevant context sentences.

Paired with the in-process FAISS store, these make `docker compose up`, the end-to-end test,
and the evaluation harness fully reproducible. Real providers (sentence-transformers, Ollama,
OpenAI, Anthropic, Qdrant, Chroma) switch on by configuration only.

## Consequences

- One-command demo, deterministic tests, and an evaluation suite that runs anywhere.
- The default answers are extractive rather than freely generated. The evaluation README is
  explicit that the lexical embedder caps retrieval quality and that switching to a real
  embedder improves it; re-run `make eval` to compare.
