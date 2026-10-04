"""Streamlit UI for the RAG service.

A thin client over the FastAPI service: it holds no RAG logic, consumes the SSE stream so the
answer appears token by token, and renders the cited sources and the retrieval/generation
latency split.

Run with:  streamlit run ui/streamlit_app.py   (set RAG_API_URL to point at the API)
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

DEFAULT_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = httpx.Timeout(60.0)


def _headers() -> dict[str, str]:
    api_key = st.session_state.get("api_key", "")
    return {"X-API-Key": api_key} if api_key else {}


def fetch_info(api_url: str) -> dict[str, Any] | None:
    try:
        response = httpx.get(f"{api_url}/api/v1/info", headers=_headers(), timeout=10.0)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data
    except httpx.HTTPError:
        return None


def ingest_sample_corpus(api_url: str) -> dict[str, Any]:
    response = httpx.post(
        f"{api_url}/api/v1/ingest",
        json={"use_sample_corpus": True},
        headers=_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def stream_answer(api_url: str, question: str, top_k: int, sink: dict[str, Any]) -> Iterator[str]:
    """Yield answer tokens from the SSE stream; collect sources/timings into ``sink``."""
    payload = {"question": question, "top_k": top_k}
    with httpx.stream(
        "POST",
        f"{api_url}/api/v1/query/stream",
        json=payload,
        headers=_headers(),
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()
        event = ""
        for line in response.iter_lines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:") :].strip())
                if event == "token":
                    yield str(data.get("text", ""))
                elif event == "sources":
                    sink["sources"] = data.get("sources", [])
                elif event == "done":
                    sink["timings"] = data.get("timings", {})
                    sink["citations"] = data.get("citations", [])
                elif event == "error":
                    sink["error"] = data.get("message", "unknown error")


def _render_sources(sources: list[dict[str, Any]], citations: list[int]) -> None:
    if not sources:
        return
    st.subheader("Sources")
    cited = set(citations)
    for source in sources:
        marker = "✅" if source["number"] in cited else "•"
        title = source.get("title") or source["doc_id"]
        with st.expander(f"{marker} [{source['number']}] {title}  ·  score {source['score']:.3f}"):
            st.write(source["snippet"])
            st.caption(f"chunk: `{source['chunk_id']}`")


def main() -> None:
    st.set_page_config(page_title="RAG System", layout="wide")
    st.session_state.setdefault("api_key", "")

    with st.sidebar:
        st.title("RAG System")
        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        st.session_state["api_key"] = st.text_input("API key (optional)", type="password")
        top_k = st.slider("Sources to retrieve (top-k)", min_value=1, max_value=10, value=4)

        st.divider()
        if st.button("Ingest sample corpus", use_container_width=True):
            try:
                result = ingest_sample_corpus(api_url)
                st.success(
                    f"Ingested {result['documents']} docs → "
                    f"{result['chunks']} chunks ({result['took_ms']} ms)"
                )
            except httpx.HTTPError as exc:
                st.error(f"Ingestion failed: {exc}")

        st.divider()
        info = fetch_info(api_url)
        if info is None:
            st.warning("API not reachable. Is the service running?")
        else:
            st.caption("**Effective configuration**")
            st.json(
                {
                    "embedding": f"{info['embedding_provider']} ({info['embedding_model']})",
                    "llm": f"{info['llm_provider']} ({info['llm_model']})",
                    "vector_store": info["vector_store"],
                    "top_k": info["top_k"],
                },
                expanded=True,
            )

    st.title("Ask a grounded question")
    st.caption(
        "Answers are generated **only** from the ingested corpus and cite their sources as `[n]`."
    )
    question = st.text_input(
        "Your question", placeholder="e.g. How many castes are there in a honeybee colony?"
    )

    if st.button("Ask", type="primary") and question.strip():
        sink: dict[str, Any] = {}
        try:
            with st.spinner("Retrieving and generating…"):
                st.subheader("Answer")
                st.write_stream(stream_answer(api_url, question, top_k, sink))
        except httpx.HTTPError as exc:
            st.error(f"Request failed: {exc}")
            return

        if sink.get("error"):
            st.error(sink["error"])
            return

        _render_sources(sink.get("sources", []), sink.get("citations", []))
        timings = sink.get("timings", {})
        if timings:
            st.caption(
                f"⏱ retrieval {timings.get('retrieval_ms', 0):.0f} ms · "
                f"generation {timings.get('generation_ms', 0):.0f} ms · "
                f"total {timings.get('total_ms', 0):.0f} ms"
            )


if __name__ == "__main__":
    main()
