# syntax=docker/dockerfile:1

# ----------------------------------------------------------------------------------------
# Builder: resolve the locked dependency set into a self-contained virtualenv with uv.
# Installs core + the `ui` and `qdrant` extras (so the single image can serve the API,
# run the Streamlit UI, and use Qdrant). Cloud providers (OpenAI/Anthropic/Ollama) work
# from the core install: they are called over httpx, with no extra SDK. The heavy `local`
# extra (sentence-transformers/torch) is intentionally omitted; add it for a local-embeddings image.
# ----------------------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra ui --extra qdrant

# Pre-fetch the tiktoken BPE table so the runtime image is fully offline-capable.
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# ----------------------------------------------------------------------------------------
# Runtime: small, non-root image with only the venv, source, sample corpus and UI.
# ----------------------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8000 \
    RAG_FAISS_PATH=/app/.data/faiss

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/.tiktoken /app/.tiktoken
COPY --chown=app:app src ./src
COPY --chown=app:app ui ./ui
COPY --chown=app:app data ./data
COPY --chown=app:app pyproject.toml README.md ./

RUN mkdir -p /app/.data && chown -R app:app /app/.data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health', timeout=3).status_code==200 else 1)"

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
