# 2. Cloud provider adapters call REST over httpx (not vendor SDKs)

- **Status:** Accepted
- **Date:** 2026-06-29

## Context

The OpenAI, Anthropic and Ollama integrations need async I/O, retries/timeouts, streaming,
and tests that never touch the network.

## Decision

Implement the cloud adapters directly against the providers' REST endpoints with a shared
async `httpx` client (`rag.adapters._http`), rather than the vendor SDKs. Retries with
exponential backoff (tenacity) and timeouts live in one place; failures map to the domain
exception hierarchy, so a `ProviderUnavailableError` becomes a 503 and a `ProviderError` a
502. Tests mock the HTTP layer with `respx`.

## Consequences

- The dependency surface stays small: cloud providers work from the base install (no SDK),
  and tests are fully deterministic via `respx`.
- We own the request/response shapes, so provider API changes must be tracked. The Anthropic
  adapter omits `temperature` (rejected by current Claude models) and pins
  `anthropic-version: 2023-06-01`; model IDs are configurable.
- We forego SDK conveniences (typed models, auto-pagination), which is acceptable for the
  small set of calls this system makes.

## Alternatives considered

The official OpenAI and Anthropic SDKs were the obvious alternative. They give typed clients
and handle pagination, but they pull in extra dependency trees and are harder to test
deterministically than mocking HTTP with `respx`. For the handful of endpoints used here,
owning the request and response shapes is a fair trade.
