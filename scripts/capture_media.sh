#!/usr/bin/env bash
# Regenerate the UI screenshots and demo GIF in docs/assets/ using the offline stack.
#
# Prerequisites (one-time):
#   uv sync --extra ui --group media
#   uv run playwright install chromium
#
# Usage:  ./scripts/capture_media.sh
set -euo pipefail

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
tmp_store="$(mktemp -d)"

export RAG_LOG_JSON=false
export RAG_RATE_LIMIT=
export RAG_FAISS_PATH="${tmp_store}/faiss"
export RAG_API_URL="http://127.0.0.1:${API_PORT}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

uv run uvicorn rag.api.main:app --host 127.0.0.1 --port "${API_PORT}" &
api_pid=$!
uv run streamlit run ui/streamlit_app.py \
    --server.port "${UI_PORT}" --server.address 127.0.0.1 --server.headless true &
ui_pid=$!
trap 'kill "${api_pid}" "${ui_pid}" 2>/dev/null || true; rm -rf "${tmp_store}"' EXIT

echo "Waiting for API…"
curl -sf --retry 30 --retry-all-errors --retry-delay 1 "http://127.0.0.1:${API_PORT}/health" >/dev/null
curl -s -X POST "http://127.0.0.1:${API_PORT}/api/v1/ingest" \
    -H 'content-type: application/json' -d '{"use_sample_corpus": true}' >/dev/null
echo "Waiting for UI…"
curl -sf --retry 40 --retry-all-errors --retry-delay 1 "http://127.0.0.1:${UI_PORT}/_stcore/health" >/dev/null

rm -rf docs/assets/_frames
uv run python scripts/capture_ui.py
echo "Done. See docs/assets/."
