# 5. SSE streaming and a pure-ASGI middleware stack

- **Status:** Accepted
- **Date:** 2026-06-29

## Context

Answers should appear token-by-token, every request needs a correlation id in logs and
responses, and probes/metrics must never be rate-limited. Starlette's `BaseHTTPMiddleware`
is known to buffer streaming responses and can break contextvar propagation.

## Decision

Stream answers with Server-Sent Events via `sse-starlette`: a `meta` event, then `sources`,
then a run of `token` events, then `done` (plus an `error` event so a mid-stream failure still
terminates cleanly). Implement request-context handling as a pure-ASGI middleware
(`X-Request-ID`, request-scoped log context, access log) so SSE keeps streaming and the bound
`request_id` propagates into endpoint logs. Rate limiting is a small pure-ASGI fixed-window
limiter scoped to `/api/v1` (probes and `/metrics` are always exempt).

## Consequences

- Real token streaming through the whole middleware chain; consistent `request_id` in logs.
- A custom limiter replaces slowapi, which mis-scoped probe endpoints and behaved
  inconsistently under the ASGI test transport. It is in-process (per replica); a
  multi-replica deployment should front it with a shared/gateway limiter.
