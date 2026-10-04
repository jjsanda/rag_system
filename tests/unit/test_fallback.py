"""Unit tests for the graceful-degradation fallback wrappers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from rag.adapters.embeddings.fake import HashingEmbedder
from rag.adapters.fallback import FallbackEmbedder, FallbackLLM
from rag.domain.exceptions import ProviderUnavailableError
from rag.domain.models import Chunk, RetrievedChunk
from rag.domain.prompt import assemble_prompt
from tests.fakes import FakeLLM


class _UnavailableLLM:
    """A primary LLM that is always unavailable."""

    @property
    def model_name(self) -> str:
        return "primary-model"

    @property
    def provider(self) -> str:
        return "primary"

    async def generate(self, *, system: str, prompt: str) -> str:
        await self._fail()
        return ""  # pragma: no cover

    async def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]:
        await self._fail()
        yield ""  # pragma: no cover

    async def _fail(self) -> None:
        raise ProviderUnavailableError("primary down", provider="primary")


class _FailsAfterFirstToken(_UnavailableLLM):
    async def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]:
        yield "partial "
        await self._fail()


class _UnavailableEmbedder:
    @property
    def model_name(self) -> str:
        return "primary-embed"

    @property
    def dimension(self) -> int:
        return 8

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise ProviderUnavailableError("primary down", provider="primary")


def _prompt_with_context() -> str:
    chunk = Chunk("d::0", "d", 0, "Bees build hexagonal comb.", 5, title="Bees")
    prompt, _ = assemble_prompt("What shape is comb?", [RetrievedChunk(chunk, 0.9)])
    return prompt


async def test_fallback_llm_generate_degrades() -> None:
    fallback = FakeLLM()
    wrapper = FallbackLLM(_UnavailableLLM(), fallback)
    answer = await wrapper.generate(system="s", prompt=_prompt_with_context())

    assert fallback.calls == 1
    assert "[1]" in answer
    assert wrapper.provider == "primary"  # identity reflects the configured primary
    assert wrapper.model_name == "primary-model"


async def test_fallback_llm_stream_degrades_before_first_token() -> None:
    wrapper = FallbackLLM(_UnavailableLLM(), FakeLLM())
    tokens = [t async for t in wrapper.stream(system="s", prompt=_prompt_with_context())]
    assert "[1]" in "".join(tokens)


async def test_fallback_llm_stream_propagates_after_first_token() -> None:
    wrapper = FallbackLLM(_FailsAfterFirstToken(), FakeLLM())
    with pytest.raises(ProviderUnavailableError):
        _ = [t async for t in wrapper.stream(system="s", prompt="p")]


async def test_fallback_embedder_degrades() -> None:
    wrapper = FallbackEmbedder(_UnavailableEmbedder(), HashingEmbedder(dim=8))
    vectors = await wrapper.embed(["hello"])
    assert len(vectors[0]) == 8
    assert wrapper.dimension == 8
    assert wrapper.model_name == "primary-embed"
