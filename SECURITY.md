# Security

## Reporting a vulnerability

Please report security issues privately to **pepek.svanda@gmail.com** rather than opening a
public issue. Include steps to reproduce and the impact; you can expect an acknowledgement
within a few days.

## Security posture

This is a portfolio project, but it is built with realistic defensive measures.

### Prompt-injection mitigation & system-prompt isolation

RAG retrieves untrusted text and feeds it to an LLM, which makes prompt injection a real
concern. The mitigations (see `rag.domain.prompt`):

- **System-prompt isolation.** Instructions live *only* in the system role. The system
  prompt tells the model to answer solely from the provided context, to treat that context as
  untrusted data rather than instructions, to never follow directives found inside it, and to
  never reveal or change the system prompt.
- **Structural separation.** Retrieved chunks are placed in a numbered, delimited context
  block, clearly separated from the user's question. Both the chunk text **and the
  (attacker-controllable) document title/label** are neutralised to a single line, so neither
  can forge a new `[n]` context block or the `Question:` boundary. The round-trip between
  `assemble_prompt` and `parse_context`/`parse_question` (including an adversarial title) is
  tested.
- **Injection detection.** `looks_like_injection` flags classic override phrasing for metrics
  (`rag_prompt_injection_flags_total`) and logging; mitigation remains structural.
- **Input limits.** `RAG_MAX_QUESTION_CHARS` and `RAG_MAX_DOCUMENT_CHARS` cap input sizes
  (rejected with `422`), backed by Pydantic `max_length` / list-size ceilings so a
  pathological body is refused during deserialization.

### Configuration & secrets

- No secrets in code. All configuration is read via `pydantic-settings`; API keys come from
  the conventional environment variables and are never logged or returned by `/api/v1/info`.
- `.env` is git-ignored; only `.env.example` (no secrets) is committed. Pre-commit runs
  `detect-private-key`.
- Optional `X-API-Key` gate on the API (`RAG_API_KEY`); per-IP rate limiting on `/api/v1`.

### Supply chain

- CI scans dependencies with `pip-audit` and the filesystem with Trivy; CodeQL runs SAST on
  every push/PR.
- The Docker image runs as a non-root user, and the build keeps the final layer small.

#### Known advisories

- `diskcache 5.6.3`: CVE-2025-69872 (no fixed release available upstream yet). Low exposure:
  `diskcache` backs only the **opt-in** disk embedding cache (`RAG_EMBEDDING_CACHE=disk`); the
  default is the in-memory cache, so the affected path is off by default. Tracked for a bump
  once a patched release ships.

### Resilience

- Provider calls have timeouts and bounded retries; on unavailability the system can degrade
  to the offline providers (`RAG_FALLBACK_TO_OFFLINE`) rather than fail the request.
