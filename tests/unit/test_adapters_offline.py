"""Unit tests for the offline-capable adapters (no network, no external services).

The single ``tiktoken`` test downloads its encoding on first use and is skipped if that is
unavailable (e.g. a fully air-gapped machine); everything else is hermetic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag.adapters.cache.disk import DiskEmbeddingCache
from rag.adapters.cache.memory import InMemoryResponseCache, LruEmbeddingCache
from rag.adapters.embeddings.caching import CachingEmbedder, embedding_cache_key
from rag.adapters.embeddings.fake import HashingEmbedder
from rag.adapters.llm.extractive import ExtractiveLLM
from rag.adapters.loaders.filesystem import FilesystemLoader
from rag.adapters.tokenizer import TiktokenTokenizer
from rag.adapters.vectorstore.faiss_store import FaissVectorStore
from rag.domain.exceptions import ConfigError, IngestionError, VectorStoreError
from rag.domain.models import Answer, Chunk, EmbeddedChunk, RetrievedChunk, Source, Timings
from rag.domain.ports import Embedder
from rag.domain.prompt import assemble_prompt, build_system_prompt

# --------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------


def test_tiktoken_roundtrip_and_count() -> None:
    try:
        tok = TiktokenTokenizer("cl100k_base")
    except ConfigError:
        pytest.skip("tiktoken encoding unavailable (offline)")
    assert tok.count("hello world") > 0
    assert tok.decode(tok.encode("hello world")) == "hello world"


def test_tiktoken_rejects_unknown_encoding() -> None:
    with pytest.raises(ConfigError):
        TiktokenTokenizer("definitely-not-an-encoding")


# --------------------------------------------------------------------------------------
# Hashing embedder
# --------------------------------------------------------------------------------------


async def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder(dim=64)
    assert emb.dimension == 64
    first = await emb.embed(["honey bees build comb"])
    second = await emb.embed(["honey bees build comb"])
    assert first == second
    assert math.isclose(math.sqrt(sum(x * x for x in first[0])), 1.0, abs_tol=1e-5)


async def test_hashing_embedder_empty_text_is_zero_vector() -> None:
    [vector] = await HashingEmbedder(dim=8).embed([""])
    assert vector == [0.0] * 8


def test_hashing_embedder_rejects_bad_dim() -> None:
    with pytest.raises(ConfigError):
        HashingEmbedder(dim=0)


# --------------------------------------------------------------------------------------
# Caching embedder
# --------------------------------------------------------------------------------------


class _CountingEmbedder:
    """Wraps an embedder and records every text it is actually asked to embed."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self.embedded: list[str] = []

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return await self._inner.embed(texts)


async def test_caching_embedder_only_embeds_misses() -> None:
    counting = _CountingEmbedder(HashingEmbedder(dim=16))
    cached = CachingEmbedder(counting, LruEmbeddingCache())

    first = await cached.embed(["a", "b"])
    second = await cached.embed(["a", "c"])  # "a" is a cache hit

    assert counting.embedded == ["a", "b", "c"]
    assert first[0] == second[0]  # identical cached vector for "a"


def test_embedding_cache_key_is_model_scoped() -> None:
    assert embedding_cache_key("m1", "x") != embedding_cache_key("m2", "x")
    assert embedding_cache_key("m1", "x") == embedding_cache_key("m1", "x")


# --------------------------------------------------------------------------------------
# Extractive LLM
# --------------------------------------------------------------------------------------


def _retrieved(chunk_id: str, doc_id: str, text: str, title: str, score: float) -> RetrievedChunk:
    chunk = Chunk(chunk_id, doc_id, 0, text, len(text.split()), title=title)
    return RetrievedChunk(chunk=chunk, score=score)


async def test_extractive_cites_the_relevant_source() -> None:
    retrieved = [
        _retrieved("a::0", "a", "Honeybees build hexagonal comb.", "Bees", 0.9),
        _retrieved("b::0", "b", "TCP is a connection oriented transport protocol.", "Net", 0.8),
    ]
    prompt, _ = assemble_prompt("Is TCP connection oriented?", retrieved)
    answer = await ExtractiveLLM().generate(system=build_system_prompt(), prompt=prompt)
    assert "[2]" in answer
    assert "TCP" in answer


async def test_extractive_handles_no_context() -> None:
    prompt, _ = assemble_prompt("anything", [])
    answer = await ExtractiveLLM().generate(system=build_system_prompt(), prompt=prompt)
    assert "[1]" not in answer
    assert "enough information" in answer


async def test_extractive_stream_reconstructs_generate() -> None:
    retrieved = [_retrieved("a::0", "a", "Honeybees build hexagonal comb.", "Bees", 0.9)]
    prompt, _ = assemble_prompt("What shape is comb?", retrieved)
    llm = ExtractiveLLM()
    full = await llm.generate(system="s", prompt=prompt)
    streamed = "".join([token async for token in llm.stream(system="s", prompt=prompt)])
    assert streamed.strip() == full.strip()


# --------------------------------------------------------------------------------------
# FAISS vector store
# --------------------------------------------------------------------------------------


def _embedded(chunk_id: str, vector: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(Chunk(chunk_id, "d", 0, "text", 1), vector)


async def test_faiss_search_ranks_by_cosine() -> None:
    store = FaissVectorStore(dimension=3)
    await store.upsert([_embedded("x", [0.0, 1.0, 0.0]), _embedded("y", [1.0, 0.0, 0.0])])
    hits = await store.search([0.0, 2.0, 0.0], top_k=2)
    assert [h.chunk.id for h in hits] == ["x", "y"]
    assert math.isclose(hits[0].score, 1.0, abs_tol=1e-5)
    assert await store.count() == 2


async def test_faiss_rejects_wrong_dimension() -> None:
    store = FaissVectorStore(dimension=3)
    with pytest.raises(VectorStoreError):
        await store.upsert([_embedded("x", [1.0, 0.0])])


async def test_faiss_reset_clears(store_dim3: FaissVectorStore) -> None:
    await store_dim3.upsert([_embedded("x", [1.0, 0.0, 0.0])])
    await store_dim3.reset()
    assert await store_dim3.count() == 0
    assert await store_dim3.search([1.0, 0.0, 0.0], top_k=1) == []


async def test_faiss_persists_across_instances(tmp_path: Path) -> None:
    store = FaissVectorStore(dimension=3, persist_path=tmp_path)
    await store.upsert([_embedded("x", [0.0, 1.0, 0.0])])

    reopened = FaissVectorStore(dimension=3, persist_path=tmp_path)
    assert await reopened.count() == 1
    hits = await reopened.search([0.0, 1.0, 0.0], top_k=1)
    assert hits[0].chunk.id == "x"


@pytest.fixture
def store_dim3() -> FaissVectorStore:
    return FaissVectorStore(dimension=3)


# --------------------------------------------------------------------------------------
# Filesystem loader
# --------------------------------------------------------------------------------------


def test_filesystem_loader_reads_titles_and_ids(tmp_path: Path) -> None:
    (tmp_path / "with_title.md").write_text("# Honey Bees\n\nBees build comb.", encoding="utf-8")
    (tmp_path / "no_title.txt").write_text("Just text, no heading.", encoding="utf-8")

    docs = {doc.id: doc for doc in FilesystemLoader().load(tmp_path)}
    assert docs["with_title"].title == "Honey Bees"
    assert docs["no_title"].title == "no_title"
    source = docs["with_title"].source
    assert source is not None and source.endswith("with_title.md")


def test_filesystem_loader_missing_path_raises() -> None:
    with pytest.raises(IngestionError):
        FilesystemLoader().load("/nope/does/not/exist")


def test_filesystem_loader_reads_sample_corpus() -> None:
    corpus = Path(__file__).resolve().parents[2] / "data" / "corpus"
    docs = FilesystemLoader().load(corpus)
    assert len(docs) >= 8
    assert all(doc.title and doc.text for doc in docs)


# --------------------------------------------------------------------------------------
# Caches
# --------------------------------------------------------------------------------------


def test_lru_embedding_cache_evicts_oldest() -> None:
    cache = LruEmbeddingCache(max_size=2)
    cache.set("a", [1.0])
    cache.set("b", [2.0])
    assert cache.get("a") == [1.0]  # touch "a" so "b" becomes the oldest
    cache.set("c", [3.0])  # evicts "b"
    assert cache.get("b") is None
    assert cache.get("a") == [1.0]
    assert cache.get("c") == [3.0]


def test_response_cache_returns_and_expires() -> None:
    cache = InMemoryResponseCache(ttl_seconds=None)
    answer = Answer(
        "x", (Source(1, "d", "d::0", "T", 0.5, "snip"),), (1,), Timings(1, 2, 3), "m", "p"
    )
    cache.set("k", answer)
    assert cache.get("k") is answer

    expiring = InMemoryResponseCache(ttl_seconds=60)
    expiring.set("k", answer)
    expiring._data["k"] = (0.0, answer)  # force an already-expired entry
    assert expiring.get("k") is None


def test_disk_embedding_cache_roundtrip(tmp_path: Path) -> None:
    cache = DiskEmbeddingCache(tmp_path / "embcache")
    assert cache.get("missing") is None
    cache.set("k", [1.0, 2.0, 3.0])
    assert cache.get("k") == [1.0, 2.0, 3.0]
    cache.close()


# --------------------------------------------------------------------------------------
# Local sentence-transformers embedder (stubbed, no torch needed)
# --------------------------------------------------------------------------------------


async def test_local_embedder_with_stub_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    import numpy as np

    class _StubModel:
        def __init__(self, name: str, device: str | None = None) -> None:
            self.name = name

        def get_sentence_embedding_dimension(self) -> int:
            return 4

        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            return np.array([[1.0, 0.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _StubModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    from rag.adapters.embeddings.local import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("stub/model")
    assert embedder.dimension == 4
    assert embedder.model_name == "stub/model"
    vectors = await embedder.embed(["a", "b"])
    assert vectors == [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    assert await embedder.embed([]) == []
