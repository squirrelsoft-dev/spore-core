"""Tests for the embedding + vector_store modules (parity plan §5.14).

Mirrors ``rust/crates/spore-core/src/embedding.rs`` and ``vector_store.rs``.
"""

from __future__ import annotations


import httpx
import pytest
import respx

from spore_core import (
    EmbeddingInfo,
    InMemoryVectorStore,
    MetadataFilter,
    OllamaEmbeddingModel,
    OpenAIEmbeddingModel,
    VectorRecord,
    VectorStoreError,
    cosine_similarity,
)
from spore_core.model import ProviderError, RateLimited


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_identical_vectors() -> None:
    assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_magnitude_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_length_mismatch_returns_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0


# ---------------------------------------------------------------------------
# OllamaEmbeddingModel
# ---------------------------------------------------------------------------


@respx.mock
async def test_ollama_embed_posts_to_api_embed() -> None:
    route = respx.post("http://localhost:11434/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]})
    )
    model = OllamaEmbeddingModel("nomic-embed-text")
    vectors = await model.embed(["hello", "world"])
    assert route.called
    body = route.calls.last.request.content.decode()
    assert "nomic-embed-text" in body
    assert "hello" in body
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@respx.mock
async def test_ollama_embed_one_returns_single_vector() -> None:
    respx.post("http://localhost:11434/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
    )
    model = OllamaEmbeddingModel("nomic")
    v = await model.embed_one("x")
    assert v == [0.1, 0.2]


async def test_ollama_embed_empty_inputs_no_request() -> None:
    model = OllamaEmbeddingModel("nomic")
    assert await model.embed([]) == []


@respx.mock
async def test_ollama_embed_non_success_is_provider_error() -> None:
    respx.post("http://localhost:11434/api/embed").mock(
        return_value=httpx.Response(500, text="boom")
    )
    model = OllamaEmbeddingModel("nomic")
    with pytest.raises(ProviderError) as ei:
        await model.embed_one("x")
    assert ei.value.code == 500


async def test_ollama_probe_dimensions() -> None:
    @respx.mock
    async def _inner() -> None:
        respx.post("http://localhost:11434/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.0, 0.0, 0.0, 0.0]]})
        )
        model = OllamaEmbeddingModel("nomic")
        d = await model.probe_dimensions()
        assert d == 4

    await _inner()


def test_ollama_info_and_dimensions() -> None:
    model = OllamaEmbeddingModel("nomic").with_dimensions(768)
    assert model.dimensions() == 768
    assert model.info() == EmbeddingInfo(provider="ollama", model_id="nomic", dimensions=768)


# ---------------------------------------------------------------------------
# OpenAIEmbeddingModel
# ---------------------------------------------------------------------------


@respx.mock
async def test_openai_embed_posts_to_embeddings_endpoint() -> None:
    route = respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.1, 0.2], "index": 0},
                    {"embedding": [0.3, 0.4], "index": 1},
                ]
            },
        )
    )
    model = OpenAIEmbeddingModel("sk-test", "text-embedding-3-small")
    vectors = await model.embed(["a", "b"])
    assert route.called
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer sk-test"
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


@respx.mock
async def test_openai_embed_sorts_by_index() -> None:
    respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.3, 0.4], "index": 1},
                    {"embedding": [0.1, 0.2], "index": 0},
                ]
            },
        )
    )
    model = OpenAIEmbeddingModel("sk-test", "text-embedding-3-small")
    vectors = await model.embed(["a", "b"])
    # Sorted by index: index 0 first.
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


@respx.mock
async def test_openai_embed_429_is_rate_limited() -> None:
    respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(429, headers={"retry-after": "2"}, text="slow down")
    )
    model = OpenAIEmbeddingModel("sk-test", "text-embedding-3-small").with_max_retries(0)
    with pytest.raises(RateLimited):
        await model.embed(["a"])


def test_openai_default_dimensions_known_families() -> None:
    assert OpenAIEmbeddingModel("k", "text-embedding-3-large").dimensions() == 3072
    assert OpenAIEmbeddingModel("k", "text-embedding-3-small").dimensions() == 1536
    assert OpenAIEmbeddingModel("k", "text-embedding-ada-002").dimensions() == 1536
    assert OpenAIEmbeddingModel("k", "unknown").dimensions() == 0


@respx.mock
async def test_openai_embed_sends_dimensions_param_for_v3() -> None:
    route = respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1], "index": 0}]})
    )
    model = OpenAIEmbeddingModel("sk-test", "text-embedding-3-small").with_dimensions(256)
    await model.embed(["a"])
    body = route.calls.last.request.content.decode()
    assert '"dimensions"' in body and "256" in body


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------


async def test_in_memory_vector_store_upsert_search_filter() -> None:
    store = InMemoryVectorStore()
    await store.add(
        [
            VectorRecord(id="a", vector=[1.0, 0.0], metadata={"kind": "chunk", "stack": "x"}),
            VectorRecord(id="b", vector=[0.0, 1.0], metadata={"kind": "summary", "stack": "x"}),
            VectorRecord(id="c", vector=[1.0, 0.1], metadata={"kind": "chunk", "stack": "y"}),
        ]
    )
    # Upsert overwrites.
    await store.add(
        [VectorRecord(id="a", vector=[1.0, 0.0], metadata={"kind": "chunk", "stack": "z"})]
    )
    hits = await store.search([1.0, 0.0], top_k=2)
    assert hits[0].id == "a"
    assert hits[0].score == pytest.approx(1.0)

    # Filter by metadata.
    filt = MetadataFilter().eq("kind", "chunk").eq("stack", "y")
    filtered = await store.search([1.0, 0.0], top_k=5, filter=filt)
    assert [h.id for h in filtered] == ["c"]


async def test_in_memory_vector_store_dimension_mismatch() -> None:
    store = InMemoryVectorStore()
    await store.add([VectorRecord(id="a", vector=[1.0, 0.0])])
    with pytest.raises(VectorStoreError) as ei:
        await store.add([VectorRecord(id="b", vector=[1.0, 0.0, 0.0])])
    assert ei.value.kind == "dimension_mismatch"
