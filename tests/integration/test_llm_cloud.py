"""Hermetic tests for the httpx-based cloud adapters using respx (no real network)."""

from __future__ import annotations

import httpx
import pytest
import respx

from rag.adapters.embeddings.openai import OpenAIEmbedder
from rag.adapters.llm.anthropic import AnthropicChatLLM
from rag.adapters.llm.ollama import OllamaChatLLM
from rag.adapters.llm.openai import OpenAIChatLLM
from rag.domain.exceptions import LLMProviderError, ProviderUnavailableError

_OPENAI_CHAT = "https://api.openai.com/v1/chat/completions"
_OPENAI_EMBED = "https://api.openai.com/v1/embeddings"
_ANTHROPIC = "https://api.anthropic.com/v1/messages"
_OLLAMA = "http://localhost:11434/api/chat"


# --------------------------------------------------------------------------------------
# OpenAI chat
# --------------------------------------------------------------------------------------


@respx.mock
async def test_openai_generate() -> None:
    respx.post(_OPENAI_CHAT).mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Hello [1]"}}]})
    )
    llm = OpenAIChatLLM(api_key="sk-test")
    assert await llm.generate(system="s", prompt="p") == "Hello [1]"
    await llm.aclose()


@respx.mock
async def test_openai_stream() -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(_OPENAI_CHAT).mock(return_value=httpx.Response(200, content=sse))
    llm = OpenAIChatLLM(api_key="sk-test")
    tokens = [t async for t in llm.stream(system="s", prompt="p")]
    assert "".join(tokens) == "Hello"
    await llm.aclose()


@respx.mock
async def test_openai_retries_then_succeeds() -> None:
    route = respx.post(_OPENAI_CHAT).mock(
        side_effect=[
            httpx.Response(429, text="slow down"),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    llm = OpenAIChatLLM(api_key="sk-test", max_retries=3, backoff_s=0.0)
    assert await llm.generate(system="s", prompt="p") == "ok"
    assert route.call_count == 2
    await llm.aclose()


@respx.mock
async def test_openai_4xx_is_not_retried() -> None:
    route = respx.post(_OPENAI_CHAT).mock(return_value=httpx.Response(400, text="bad request"))
    llm = OpenAIChatLLM(api_key="sk-test", max_retries=3, backoff_s=0.0)
    with pytest.raises(LLMProviderError):
        await llm.generate(system="s", prompt="p")
    assert route.call_count == 1
    await llm.aclose()


@respx.mock
async def test_openai_exhausted_retries_is_unavailable() -> None:
    respx.post(_OPENAI_CHAT).mock(return_value=httpx.Response(503, text="overloaded"))
    llm = OpenAIChatLLM(api_key="sk-test", max_retries=2, backoff_s=0.0)
    with pytest.raises(ProviderUnavailableError):
        await llm.generate(system="s", prompt="p")
    await llm.aclose()


# --------------------------------------------------------------------------------------
# OpenAI embeddings
# --------------------------------------------------------------------------------------


@respx.mock
async def test_openai_embeddings_preserve_order() -> None:
    respx.post(_OPENAI_EMBED).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            },
        )
    )
    emb = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-small")
    vectors = await emb.embed(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]  # re-ordered by `index`
    assert emb.dimension == 1536
    await emb.aclose()


# --------------------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------------------


@respx.mock
async def test_anthropic_generate_joins_text_blocks() -> None:
    respx.post(_ANTHROPIC).mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "Bees build "},
                    {"type": "text", "text": "comb [1]"},
                ],
                "stop_reason": "end_turn",
            },
        )
    )
    llm = AnthropicChatLLM(api_key="sk-ant")
    assert await llm.generate(system="s", prompt="p") == "Bees build comb [1]"
    await llm.aclose()


@respx.mock
async def test_anthropic_stream_text_deltas() -> None:
    sse = (
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Bee"}}\n\n'
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"s"}}\n\n'
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
    respx.post(_ANTHROPIC).mock(return_value=httpx.Response(200, content=sse))
    llm = AnthropicChatLLM(api_key="sk-ant")
    tokens = [t async for t in llm.stream(system="s", prompt="p")]
    assert "".join(tokens) == "Bees"
    await llm.aclose()


# --------------------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------------------


@respx.mock
async def test_ollama_generate() -> None:
    respx.post(_OLLAMA).mock(
        return_value=httpx.Response(200, json={"message": {"content": "Hi [1]"}, "done": True})
    )
    llm = OllamaChatLLM()
    assert await llm.generate(system="s", prompt="p") == "Hi [1]"
    await llm.aclose()


@respx.mock
async def test_ollama_stream_jsonl() -> None:
    jsonl = (
        '{"message":{"content":"Hi "},"done":false}\n'
        '{"message":{"content":"there"},"done":true}\n'
    )
    respx.post(_OLLAMA).mock(return_value=httpx.Response(200, content=jsonl))
    llm = OllamaChatLLM()
    tokens = [t async for t in llm.stream(system="s", prompt="p")]
    assert "".join(tokens) == "Hi there"
    await llm.aclose()
