"""MemoryProvider — persist and retrieve knowledge across turns and sessions
(issue #8).

Mirrors the Rust reference at ``rust/crates/spore-core/src/memory.rs``.

Stores two distinct kinds of memory:

* **Episodic**: what happened during a specific session, created by the
  harness from session observations.
* **Semantic**: generalized knowledge — skills, rules, patterns, domain
  facts — distilled from one or more episodic traces.

See ``docs/harness-engineering-concepts.md`` § "MemoryProvider" for the rules
this module enforces. The reference implementation
:class:`StandardMemoryProvider` is in-memory; production deployments swap in
a durable backing store without changing the protocol surface.

Rules enforced:

* Episodic and semantic memory live in separate stores.
* :attr:`MergeStrategy.REPLACE` archives the previous record under a fresh
  archival id, links it into ``previous_versions``, and bumps ``version``.
  Hard deletes are not permitted.
* :attr:`MergeStrategy.APPEND` concatenates content into the existing record
  in place — same id, no new version. Use only for accumulator memories.
* :attr:`MergeStrategy.REJECT` raises :class:`MergeConflict` on collision.
* ``MetaAgentProposed`` memories are forced into
  :class:`MemoryStatusPendingReview` regardless of caller-supplied status.
* Empty content fails with :class:`ValidationFailed`.
* :meth:`MemoryProvider.query` returns items with
  ``relevance_score >= query.min_relevance``, capped at ``max_items``,
  sorted by score descending. Only ``Active`` semantic memories are
  returned.
"""

from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from enum import Enum
from typing import (
    Annotated,
    ClassVar,
    Literal,
    NewType,
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field

from .errors import SporeError
from .harness import SessionId

# ============================================================================
# Identity & time
# ============================================================================

MemoryId = NewType("MemoryId", str)
"""Stable identifier for a memory record."""

Timestamp = NewType("Timestamp", str)
"""RFC 3339 / ISO 8601 timestamp, stored as a string for cross-language
fixture portability."""


def new_memory_id(s: str) -> MemoryId:
    return MemoryId(s)


def new_timestamp(s: str) -> Timestamp:
    return Timestamp(s)


# ============================================================================
# Pydantic base
# ============================================================================


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ============================================================================
# MemorySource (discriminated union on ``kind``)
# ============================================================================


class MemorySourceManual(_Model):
    kind: Literal["manual"] = "manual"


class MemorySourceSessionGenerated(_Model):
    kind: Literal["session_generated"] = "session_generated"
    session_id: SessionId


class MemorySourceTraceDistilled(_Model):
    kind: Literal["trace_distilled"] = "trace_distilled"
    session_ids: list[SessionId] = Field(default_factory=list)


class MemorySourceMetaAgentProposed(_Model):
    kind: Literal["meta_agent_proposed"] = "meta_agent_proposed"
    approved_by: str | None = None


MemorySource = Annotated[
    MemorySourceManual
    | MemorySourceSessionGenerated
    | MemorySourceTraceDistilled
    | MemorySourceMetaAgentProposed,
    Field(discriminator="kind"),
]


# ============================================================================
# MemoryStatus (discriminated union on ``kind``)
# ============================================================================


class MemoryStatusActive(_Model):
    kind: Literal["active"] = "active"


class MemoryStatusDeprecated(_Model):
    kind: Literal["deprecated"] = "deprecated"
    reason: str
    at: Timestamp


class MemoryStatusPendingReview(_Model):
    kind: Literal["pending_review"] = "pending_review"
    proposed_at: Timestamp


MemoryStatus = Annotated[
    MemoryStatusActive | MemoryStatusDeprecated | MemoryStatusPendingReview,
    Field(discriminator="kind"),
]


# ============================================================================
# Records
# ============================================================================


class EpisodicMemory(_Model):
    id: MemoryId
    session_id: SessionId
    content: str
    created_at: Timestamp
    tags: list[str] = Field(default_factory=list)


class SemanticMemory(_Model):
    id: MemoryId
    content: str
    source: MemorySource
    domain: str | None = None
    version: int = 1
    previous_versions: list[MemoryId] = Field(default_factory=list)
    created_at: Timestamp
    updated_at: Timestamp
    status: MemoryStatus


class MemoryItem(_Model):
    """Scored query result returned by :meth:`MemoryProvider.query`.

    This is the canonical ``MemoryItem`` definition for issue #8; the older
    forward-declared stub in :mod:`spore_core.context` re-exports this type.
    """

    memory: SemanticMemory
    relevance_score: float


class MemoryQuery(_Model):
    task_instruction: str
    domain: str | None = None
    session_id: SessionId | None = None
    min_relevance: float = 0.5
    max_items: int = 10


class MergeStrategy(str, Enum):
    REPLACE = "replace"
    APPEND = "append"
    REJECT = "reject"


# ============================================================================
# Errors
# ============================================================================


class MemoryError(SporeError):
    """Root of every error raised by a :class:`MemoryProvider`."""

    kind: ClassVar[str] = "MemoryError"


class NotFound(MemoryError):
    kind: ClassVar[str] = "NotFound"

    def __init__(self, id: MemoryId) -> None:
        self.id = id
        super().__init__(f"memory not found: {id!r}")


class MergeConflict(MemoryError):
    kind: ClassVar[str] = "MergeConflict"

    def __init__(self, existing: MemoryId, reason: str) -> None:
        self.existing = existing
        self.reason = reason
        super().__init__(f"merge conflict on {existing!r}: {reason}")


class ValidationFailed(MemoryError):
    kind: ClassVar[str] = "ValidationFailed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"validation failed: {reason}")


class StorageError(MemoryError):
    kind: ClassVar[str] = "StorageError"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"storage error: {reason}")


# ============================================================================
# Protocol
# ============================================================================


@runtime_checkable
class MemoryProvider(Protocol):
    """Canonical issue-#8 ``MemoryProvider`` interface."""

    # ── Episodic ────────────────────────────────────────────────────────────
    async def store_episodic(self, memory: EpisodicMemory) -> MemoryId: ...

    async def get_episodic(self, session_id: SessionId) -> list[EpisodicMemory]: ...

    # ── Semantic ────────────────────────────────────────────────────────────
    async def store_semantic(
        self, memory: SemanticMemory, on_conflict: MergeStrategy
    ) -> MemoryId: ...

    async def get_semantic(self, id: MemoryId) -> SemanticMemory: ...

    async def query(self, query: MemoryQuery) -> list[MemoryItem]: ...

    # ── Lifecycle ───────────────────────────────────────────────────────────
    async def deprecate(self, id: MemoryId, reason: str) -> None: ...

    async def get_version_history(self, id: MemoryId) -> list[SemanticMemory]: ...

    async def mark_pending_review(self, id: MemoryId) -> None: ...


# ============================================================================
# Helpers
# ============================================================================


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower())


def _jaccard(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta and not tb:
        return 0.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def _validate_content(s: str) -> None:
    if not s.strip():
        raise ValidationFailed("content must not be empty")


def _enforce_meta_agent_review(memory: SemanticMemory) -> SemanticMemory:
    """Force ``MetaAgentProposed`` memories into ``PendingReview`` regardless
    of caller-supplied status. ``proposed_at`` falls back to ``created_at``
    when the caller did not supply a review timestamp."""

    if isinstance(memory.source, MemorySourceMetaAgentProposed) and not isinstance(
        memory.status, MemoryStatusPendingReview
    ):
        return memory.model_copy(
            update={
                "status": MemoryStatusPendingReview(proposed_at=memory.created_at),
            }
        )
    return memory


# ============================================================================
# StandardMemoryProvider — reference in-memory implementation
# ============================================================================


class StandardMemoryProvider:
    """Reference :class:`MemoryProvider`. In-memory; suitable for tests and
    short-lived processes.

    Relevance scoring: token-overlap (Jaccard) between the query's
    ``task_instruction`` and each candidate memory's ``content``. This is
    intentionally simple — the spec calls for embedding similarity in
    production. The protocol does not mandate the scoring algorithm, only
    that items below ``min_relevance`` are not returned.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # Insertion-ordered to preserve append order for episodic.
        self._episodic: OrderedDict[SessionId, list[EpisodicMemory]] = OrderedDict()
        self._semantic: dict[MemoryId, SemanticMemory] = {}
        self._archive_seq: int = 0

    def _next_archive_id(self, original: MemoryId) -> MemoryId:
        self._archive_seq += 1
        return MemoryId(f"{original}@v{self._archive_seq}")

    # ── Episodic ────────────────────────────────────────────────────────────

    async def store_episodic(self, memory: EpisodicMemory) -> MemoryId:
        _validate_content(memory.content)
        async with self._lock:
            self._episodic.setdefault(memory.session_id, []).append(memory)
            return memory.id

    async def get_episodic(self, session_id: SessionId) -> list[EpisodicMemory]:
        async with self._lock:
            return list(self._episodic.get(session_id, []))

    # ── Semantic ────────────────────────────────────────────────────────────

    async def store_semantic(self, memory: SemanticMemory, on_conflict: MergeStrategy) -> MemoryId:
        _validate_content(memory.content)
        memory = _enforce_meta_agent_review(memory)

        async with self._lock:
            id_ = memory.id
            existing = self._semantic.get(id_)

            if existing is not None:
                if on_conflict is MergeStrategy.REJECT:
                    raise MergeConflict(
                        existing=existing.id,
                        reason="memory exists; on_conflict=Reject",
                    )
                if on_conflict is MergeStrategy.APPEND:
                    merged = existing.model_copy(
                        update={
                            "content": existing.content + memory.content,
                            "updated_at": memory.updated_at,
                        }
                    )
                    self._semantic[id_] = merged
                    return id_
                if on_conflict is MergeStrategy.REPLACE:
                    archive_id = self._next_archive_id(existing.id)
                    archived = existing.model_copy(update={"id": archive_id})
                    self._semantic[archive_id] = archived

                    new_prev = list(existing.previous_versions)
                    new_prev.append(archive_id)
                    memory = memory.model_copy(
                        update={
                            "version": existing.version + 1,
                            "previous_versions": new_prev,
                        }
                    )

            self._semantic[id_] = memory
            return id_

    async def get_semantic(self, id: MemoryId) -> SemanticMemory:
        async with self._lock:
            m = self._semantic.get(id)
            if m is None:
                raise NotFound(id)
            return m

    async def query(self, query: MemoryQuery) -> list[MemoryItem]:
        async with self._lock:
            candidates = list(self._semantic.values())

        scored: list[MemoryItem] = []
        for m in candidates:
            if not isinstance(m.status, MemoryStatusActive):
                continue
            # Domain filter: ``Some(d)`` only matches ``Some(d)``;
            # ``Some(d)`` skips ``None``.
            if query.domain is not None and m.domain != query.domain:
                continue
            score = _jaccard(query.task_instruction, m.content)
            if score < query.min_relevance:
                continue
            scored.append(MemoryItem(memory=m, relevance_score=score))

        scored.sort(key=lambda item: item.relevance_score, reverse=True)
        if query.max_items >= 0:
            scored = scored[: query.max_items]
        return scored

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def deprecate(self, id: MemoryId, reason: str) -> None:
        async with self._lock:
            m = self._semantic.get(id)
            if m is None:
                raise NotFound(id)
            self._semantic[id] = m.model_copy(
                update={
                    "status": MemoryStatusDeprecated(reason=reason, at=m.updated_at),
                }
            )

    async def get_version_history(self, id: MemoryId) -> list[SemanticMemory]:
        async with self._lock:
            head = self._semantic.get(id)
            if head is None:
                raise NotFound(id)
            chain: list[SemanticMemory] = [head]
            for prev_id in head.previous_versions:
                prev = self._semantic.get(prev_id)
                if prev is not None:
                    chain.append(prev)
            return chain

    async def mark_pending_review(self, id: MemoryId) -> None:
        async with self._lock:
            m = self._semantic.get(id)
            if m is None:
                raise NotFound(id)
            self._semantic[id] = m.model_copy(
                update={
                    "status": MemoryStatusPendingReview(proposed_at=m.updated_at),
                }
            )


__all__ = [
    "EpisodicMemory",
    "MemoryError",
    "MemoryId",
    "MemoryItem",
    "MemoryProvider",
    "MemoryQuery",
    "MemorySource",
    "MemorySourceManual",
    "MemorySourceMetaAgentProposed",
    "MemorySourceSessionGenerated",
    "MemorySourceTraceDistilled",
    "MemoryStatus",
    "MemoryStatusActive",
    "MemoryStatusDeprecated",
    "MemoryStatusPendingReview",
    "MergeConflict",
    "MergeStrategy",
    "NotFound",
    "SemanticMemory",
    "StandardMemoryProvider",
    "StorageError",
    "Timestamp",
    "ValidationFailed",
    "new_memory_id",
    "new_timestamp",
]
