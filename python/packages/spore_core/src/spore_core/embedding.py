"""``EmbeddingModel`` — boundary between the RAG layer and an embedding provider.

Mirrors ``rust/crates/spore-core/src/embedding.rs``. The protocol deals ONLY in
text → vector; storing and searching the resulting vectors is
:mod:`spore_core.vector_store`.

Two providers ship in-core because they add no new dependency — they reuse
``httpx`` (and, for OpenAI, the shared retry/backoff plumbing):

* :class:`OllamaEmbeddingModel` — POST ``/api/embed``.
* :class:`OpenAIEmbeddingModel` — POST ``${base_url}/embeddings``.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from .model import (
    ProviderError,
    RateLimited,
    TimeoutError as ModelTimeoutError,
    Transport,
)
import json as _json

__all__ = [
    "EmbeddingInfo",
    "EmbeddingModel",
    "OllamaEmbeddingModel",
    "OpenAIEmbeddingModel",
    "cosine_similarity",
]


# ============================================================================
# Identity + cosine similarity
# ============================================================================


@dataclass(frozen=True)
class EmbeddingInfo:
    """Identity of an embedding provider/model, for routing and diagnostics."""

    provider: str
    model_id: str
    dimensions: int = 0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors in ``[-1, 1]`` (higher =
    closer). Returns ``0.0`` when either vector has zero magnitude or lengths
    differ — a degenerate input is treated as "no similarity" rather than
    ``NaN``/panic. Mirrors Rust's ``cosine_similarity``."""
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))


# ============================================================================
# Protocol
# ============================================================================


@runtime_checkable
class EmbeddingModel(Protocol):
    """Turns text into embedding vectors."""

    async def embed(self, inputs: list[str]) -> list[list[float]]: ...

    async def embed_one(self, input: str) -> list[float]: ...

    def dimensions(self) -> int: ...

    def info(self) -> EmbeddingInfo: ...


# ============================================================================
# Ollama
# ============================================================================


class OllamaEmbeddingModel:
    """``EmbeddingModel`` over a local Ollama server's ``/api/embed`` endpoint.

    No API key; default ``base_url`` is ``http://localhost:11434``. Fails fast
    on connection errors (no retry) with a "not running" hint. Ollama does not
    report vector length except in the embedding itself, so ``dimensions``
    starts at ``0`` (unknown) — set it with :meth:`with_dimensions` or discover
    it with :meth:`probe_dimensions`.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.base_url = self.DEFAULT_BASE_URL
        self._dimensions = 0
        self.timeout = self.DEFAULT_TIMEOUT
        self._client: httpx.AsyncClient | None = None

    def with_base_url(self, base_url: str) -> OllamaEmbeddingModel:
        self.base_url = base_url
        return self

    def with_dimensions(self, dimensions: int) -> OllamaEmbeddingModel:
        self._dimensions = dimensions
        return self

    def with_timeout(self, timeout: float) -> OllamaEmbeddingModel:
        self.timeout = timeout
        return self

    def with_http_client(self, client: httpx.AsyncClient) -> OllamaEmbeddingModel:
        self._client = client
        return self

    async def probe_dimensions(self) -> int:
        """Discover the model's vector length by embedding a probe string."""
        vector = await self.embed_one("dimension probe")
        return len(vector)

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        url = f"{self.base_url}/api/embed"
        body: dict[str, Any] = {"model": self.model_id, "input": inputs}
        client = self._ensure_client()
        try:
            resp = await client.post(
                url, json=body, headers={"content-type": "application/json"}, timeout=self.timeout
            )
        except httpx.TimeoutException as e:
            raise ModelTimeoutError() from e
        except httpx.ConnectError as e:
            raise ProviderError(code=0, message=f"Ollama not running at {self.base_url}") from e
        except httpx.HTTPError as e:
            raise Transport(f"HTTP transport error: {e}") from e
        if resp.status_code >= 400:
            code = resp.status_code
            text = resp.text[:500]
            raise ProviderError(code=code, message=text)
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderError(code=0, message=f"decode embed response failed: {e}") from e
        embeddings = data.get("embeddings", [])
        return [list(v) for v in embeddings]

    async def embed_one(self, input: str) -> list[float]:
        vectors = await self.embed([input])
        if not vectors:
            raise ProviderError(code=0, message="embedding provider returned no vectors")
        return vectors[0]

    def dimensions(self) -> int:
        return self._dimensions

    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(provider="ollama", model_id=self.model_id, dimensions=self._dimensions)


# ============================================================================
# OpenAI
# ============================================================================


_RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


def _backoff_delay(attempt: int) -> float:
    base = 0.5 * (1 << min(attempt, 6))
    return min(base, 30.0)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _openai_error_message(body: str) -> str:
    """Pull the message from the typed ``{error:{message}}`` envelope, falling
    back to the first 500 chars of the raw body. Mirrors Rust's
    ``provider_http::error_parts``."""
    try:
        parsed = _json.loads(body)
    except (ValueError, TypeError):
        return body[:500]
    if isinstance(parsed, dict):
        err = parsed.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str):
                return msg
    return body[:500]


class OpenAIEmbeddingModel:
    """``EmbeddingModel`` over ``${base_url}/embeddings`` (OpenAI +
    OpenAI-compatible servers). Reuses the shared retry/backoff plumbing.

    ``dimensions`` defaults from the model id for the known
    ``text-embedding-*`` families; :meth:`with_dimensions` overrides it AND, for
    the v3 families that accept the truncation parameter, asks the API for that
    length.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_TIMEOUT = 120.0
    DEFAULT_MAX_RETRIES = 3

    def __init__(self, api_key: str, model_id: str) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = self.DEFAULT_BASE_URL
        self._dimensions = self._default_dimensions(model_id)
        self.timeout = self.DEFAULT_TIMEOUT
        self.max_retries = self.DEFAULT_MAX_RETRIES
        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def from_env(env_var: str, model_id: str) -> OpenAIEmbeddingModel:
        key = os.environ.get(env_var)
        if not key:
            raise ProviderError(code=0, message=f"missing env var: {env_var}")
        return OpenAIEmbeddingModel(key, model_id)

    def with_base_url(self, base_url: str) -> OpenAIEmbeddingModel:
        self.base_url = base_url
        return self

    def with_dimensions(self, dimensions: int) -> OpenAIEmbeddingModel:
        self._dimensions = dimensions
        return self

    def with_timeout(self, timeout: float) -> OpenAIEmbeddingModel:
        self.timeout = timeout
        return self

    def with_max_retries(self, n: int) -> OpenAIEmbeddingModel:
        self.max_retries = n
        return self

    def with_http_client(self, client: httpx.AsyncClient) -> OpenAIEmbeddingModel:
        self._client = client
        return self

    @staticmethod
    def _default_dimensions(model_id: str) -> int:
        if model_id == "text-embedding-3-large":
            return 3072
        if model_id in ("text-embedding-3-small", "text-embedding-ada-002"):
            return 1536
        return 0

    def _supports_dimensions_param(self) -> bool:
        return self.model_id.startswith("text-embedding-3")

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def _send_with_retry(self, url: str, body: dict[str, Any]) -> httpx.Response:
        client = self._ensure_client()
        attempt = 0
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        payload = _json.dumps(body).encode("utf-8")
        while True:
            try:
                resp = await client.post(
                    url, content=payload, headers=headers, timeout=self.timeout
                )
            except httpx.TimeoutException as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(_backoff_delay(attempt))
                    attempt += 1
                    continue
                raise ModelTimeoutError() from e
            except httpx.HTTPError as e:
                raise Transport(f"HTTP transport error: {e}") from e
            status = resp.status_code
            if 200 <= status < 300:
                return resp
            if status in _RETRYABLE_STATUSES and attempt < self.max_retries:
                retry_after = _parse_retry_after(resp.headers.get("retry-after"))
                delay = retry_after if retry_after is not None else _backoff_delay(attempt)
                await resp.aclose()
                await asyncio.sleep(delay)
                attempt += 1
                continue
            retry_after = _parse_retry_after(resp.headers.get("retry-after"))
            text = resp.text
            await resp.aclose()
            if status == 429:
                raise RateLimited(retry_after=retry_after)
            raise ProviderError(code=status, message=_openai_error_message(text))

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        url = f"{self.base_url}/embeddings"
        dimensions = (
            self._dimensions if self._dimensions > 0 and self._supports_dimensions_param() else None
        )
        body: dict[str, Any] = {"model": self.model_id, "input": inputs}
        if dimensions is not None:
            body["dimensions"] = dimensions
        resp = await self._send_with_retry(url, body)
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderError(code=0, message=f"decode embeddings response failed: {e}") from e
        finally:
            await resp.aclose()
        # `index` is authoritative — sort by it so a non-conforming compatible
        # server can't scramble the input↔vector correspondence.
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        return [list(d["embedding"]) for d in items]

    async def embed_one(self, input: str) -> list[float]:
        vectors = await self.embed([input])
        if not vectors:
            raise ProviderError(code=0, message="embedding provider returned no vectors")
        return vectors[0]

    def dimensions(self) -> int:
        return self._dimensions

    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(provider="openai", model_id=self.model_id, dimensions=self._dimensions)
