"""``VectorStore`` — store and similarity-search embedding vectors.

Mirrors ``rust/crates/spore-core/src/vector_store.rs``. The companion to
:mod:`spore_core.embedding`: the embedder turns text into vectors, the store
persists them with arbitrary JSON metadata and answers top-k nearest-neighbour
queries. The protocol is deliberately vector-only.

An in-memory cosine implementation (:class:`InMemoryVectorStore`) ships here as
the trait's executable contract and the reference backend; the sqlite-vec
backend lives separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .embedding import cosine_similarity

__all__ = [
    "InMemoryVectorStore",
    "MetadataFilter",
    "VectorHit",
    "VectorRecord",
    "VectorStore",
    "VectorStoreError",
]


@dataclass
class VectorRecord:
    """A vector to store: a caller-chosen ``id``, the ``vector`` itself, and
    arbitrary JSON ``metadata``. Re-adding an existing ``id`` is an upsert."""

    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorHit:
    """One search result: the stored ``id``, its ``score`` (HIGHER = more
    similar), and the record's ``metadata``."""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetadataFilter:
    """An AND-ed set of equality predicates over top-level metadata keys."""

    equals: list[tuple[str, Any]] = field(default_factory=list)

    def eq(self, key: str, value: Any) -> MetadataFilter:
        self.equals.append((key, value))
        return self

    def is_empty(self) -> bool:
        return not self.equals

    def matches(self, metadata: dict[str, Any]) -> bool:
        return all(metadata.get(k) == v for k, v in self.equals)


class VectorStoreError(Exception):
    """Vector store failure."""

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        self.message = message
        super().__init__(f"{kind}: {message}")


def _dimension_mismatch(expected: int, actual: int) -> VectorStoreError:
    return VectorStoreError(
        "dimension_mismatch", f"vector dimension mismatch: expected {expected}, got {actual}"
    )


@runtime_checkable
class VectorStore(Protocol):
    """Store and similarity-search embedding vectors."""

    async def add(self, records: list[VectorRecord]) -> None: ...

    async def search(
        self, query: list[float], top_k: int, filter: MetadataFilter | None = None
    ) -> list[VectorHit]: ...

    async def delete(self, filter: MetadataFilter) -> int: ...

    async def len(self) -> int: ...

    async def is_empty(self) -> bool: ...


class InMemoryVectorStore:
    """Reference in-memory cosine :class:`VectorStore`.

    Rejects a record whose vector length differs from the store's configured
    dimensionality (``0`` = accept any length, pinning the first record's
    length)."""

    def __init__(self, dimensions: int = 0) -> None:
        self._dimensions = dimensions
        self._records: dict[str, VectorRecord] = {}

    async def add(self, records: list[VectorRecord]) -> None:
        for rec in records:
            if self._dimensions == 0:
                self._dimensions = len(rec.vector)
            elif len(rec.vector) != self._dimensions:
                raise _dimension_mismatch(self._dimensions, len(rec.vector))
            self._records[rec.id] = rec

    async def search(
        self, query: list[float], top_k: int, filter: MetadataFilter | None = None
    ) -> list[VectorHit]:
        if self._dimensions != 0 and len(query) != self._dimensions:
            raise _dimension_mismatch(self._dimensions, len(query))
        scored: list[VectorHit] = []
        for rec in self._records.values():
            if filter is not None and not filter.is_empty() and not filter.matches(rec.metadata):
                continue
            score = cosine_similarity(query, rec.vector)
            scored.append(VectorHit(id=rec.id, score=score, metadata=dict(rec.metadata)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    async def delete(self, filter: MetadataFilter) -> int:
        """Delete every record whose metadata matches ``filter``. An empty
        filter matches all records (a full clear). Returns the count removed."""
        if filter.is_empty():
            n = len(self._records)
            self._records.clear()
            return n
        victims = [rid for rid, rec in self._records.items() if filter.matches(rec.metadata)]
        for rid in victims:
            del self._records[rid]
        return len(victims)

    async def len(self) -> int:
        return len(self._records)

    async def is_empty(self) -> bool:
        return not self._records
