"""Tests for :class:`StandardMemoryProvider` — issue #8.

Mirrors the Rust unit tests in
``rust/crates/spore-core/src/memory.rs`` ``tests`` module.
"""

from __future__ import annotations

import pytest

from spore_core.harness import SessionId
from spore_core.memory import (
    EpisodicMemory,
    MemoryId,
    MemoryQuery,
    MemorySourceManual,
    MemorySourceMetaAgentProposed,
    MemoryStatusActive,
    MemoryStatusDeprecated,
    MemoryStatusPendingReview,
    MergeConflict,
    MergeStrategy,
    NotFound,
    SemanticMemory,
    StandardMemoryProvider,
    Timestamp,
    ValidationFailed,
)


def ts(s: str) -> Timestamp:
    return Timestamp(s)


def sem(id_: str, content: str) -> SemanticMemory:
    return SemanticMemory(
        id=MemoryId(id_),
        content=content,
        source=MemorySourceManual(),
        domain=None,
        version=1,
        previous_versions=[],
        created_at=ts("2026-05-16T00:00:00Z"),
        updated_at=ts("2026-05-16T00:00:00Z"),
        status=MemoryStatusActive(),
    )


def epi(id_: str, session: str, content: str) -> EpisodicMemory:
    return EpisodicMemory(
        id=MemoryId(id_),
        session_id=SessionId(session),
        content=content,
        created_at=ts("2026-05-16T00:00:00Z"),
        tags=[],
    )


# ── Rule: Episodic and semantic stored/retrieved separately ─────────────


@pytest.mark.asyncio
async def test_episodic_and_semantic_use_separate_stores() -> None:
    mp = StandardMemoryProvider()
    await mp.store_episodic(epi("e1", "s1", "ran tests"))
    await mp.store_semantic(sem("g1", "always run tests"), MergeStrategy.REJECT)

    eps = await mp.get_episodic(SessionId("s1"))
    assert len(eps) == 1
    # Episodic id should not be retrievable as semantic.
    with pytest.raises(NotFound):
        await mp.get_semantic(MemoryId("e1"))
    # Semantic id should not appear under any session's episodics.
    assert await mp.get_episodic(SessionId("g1")) == []


# ── Rule: Replace creates a new version, retains previous ───────────────


@pytest.mark.asyncio
async def test_replace_archives_previous_and_bumps_version() -> None:
    mp = StandardMemoryProvider()
    await mp.store_semantic(sem("g1", "v1 content"), MergeStrategy.REJECT)

    v2 = sem("g1", "v2 content")
    # Caller version is ignored; provider bumps from existing.
    v2 = v2.model_copy(update={"version": 1})
    await mp.store_semantic(v2, MergeStrategy.REPLACE)

    current = await mp.get_semantic(MemoryId("g1"))
    assert current.content == "v2 content"
    assert current.version == 2
    assert len(current.previous_versions) == 1

    history = await mp.get_version_history(MemoryId("g1"))
    assert len(history) == 2
    assert history[0].content == "v2 content"
    assert history[1].content == "v1 content"


@pytest.mark.asyncio
async def test_replace_chains_versions_across_multiple_updates() -> None:
    mp = StandardMemoryProvider()
    await mp.store_semantic(sem("g1", "v1"), MergeStrategy.REJECT)
    await mp.store_semantic(sem("g1", "v2"), MergeStrategy.REPLACE)
    await mp.store_semantic(sem("g1", "v3"), MergeStrategy.REPLACE)
    cur = await mp.get_semantic(MemoryId("g1"))
    assert cur.version == 3
    history = await mp.get_version_history(MemoryId("g1"))
    assert len(history) == 3


# ── Rule: Reject returns MergeConflict ──────────────────────────────────


@pytest.mark.asyncio
async def test_reject_on_conflict_errors() -> None:
    mp = StandardMemoryProvider()
    await mp.store_semantic(sem("g1", "first"), MergeStrategy.REJECT)
    with pytest.raises(MergeConflict):
        await mp.store_semantic(sem("g1", "second"), MergeStrategy.REJECT)
    # Original untouched.
    current = await mp.get_semantic(MemoryId("g1"))
    assert current.content == "first"


# ── Rule: Append concatenates in place, no new version ──────────────────


@pytest.mark.asyncio
async def test_append_concatenates_without_new_version() -> None:
    mp = StandardMemoryProvider()
    await mp.store_semantic(sem("g1", "a"), MergeStrategy.REJECT)
    await mp.store_semantic(sem("g1", "b"), MergeStrategy.APPEND)
    cur = await mp.get_semantic(MemoryId("g1"))
    assert cur.content == "ab"
    assert cur.version == 1
    assert cur.previous_versions == []


# ── Rule: Writes validated (empty content rejected) ─────────────────────


@pytest.mark.asyncio
async def test_empty_content_fails_validation() -> None:
    mp = StandardMemoryProvider()
    with pytest.raises(ValidationFailed):
        await mp.store_semantic(sem("g1", "   "), MergeStrategy.REJECT)
    with pytest.raises(ValidationFailed):
        await mp.store_episodic(epi("e1", "s1", ""))


# ── Rule: MetaAgentProposed forced to PendingReview ─────────────────────


@pytest.mark.asyncio
async def test_meta_agent_memories_forced_to_pending_review() -> None:
    mp = StandardMemoryProvider()
    m = sem("g1", "proposed skill").model_copy(
        update={
            "source": MemorySourceMetaAgentProposed(approved_by=None),
            # Caller dishonestly sets Active.
            "status": MemoryStatusActive(),
        }
    )
    await mp.store_semantic(m, MergeStrategy.REJECT)
    stored = await mp.get_semantic(MemoryId("g1"))
    assert isinstance(stored.status, MemoryStatusPendingReview)


# ── Rule: query returns scored items, filtered by min_relevance ─────────


@pytest.mark.asyncio
async def test_query_scores_filters_and_sorts() -> None:
    mp = StandardMemoryProvider()
    await mp.store_semantic(sem("g1", "rust async tokio runtime"), MergeStrategy.REJECT)
    await mp.store_semantic(sem("g2", "python pytest fixtures"), MergeStrategy.REJECT)
    await mp.store_semantic(sem("g3", "unrelated cooking recipe"), MergeStrategy.REJECT)

    q = MemoryQuery(
        task_instruction="rust tokio async",
        min_relevance=0.1,
        max_items=10,
        domain=None,
        session_id=None,
    )
    res = await mp.query(q)
    assert res
    assert res[0].memory.id == MemoryId("g1")
    # Sorted descending.
    for a, b in zip(res, res[1:]):
        assert a.relevance_score >= b.relevance_score


@pytest.mark.asyncio
async def test_query_excludes_deprecated_memories() -> None:
    mp = StandardMemoryProvider()
    await mp.store_semantic(sem("g1", "rust tokio"), MergeStrategy.REJECT)
    await mp.deprecate(MemoryId("g1"), "obsolete")
    res = await mp.query(
        MemoryQuery(
            task_instruction="rust tokio",
            min_relevance=0.0,
            max_items=10,
            domain=None,
            session_id=None,
        )
    )
    assert res == []


@pytest.mark.asyncio
async def test_query_respects_min_relevance_and_max_items() -> None:
    mp = StandardMemoryProvider()
    for i in range(5):
        await mp.store_semantic(sem(f"g{i}", f"alpha beta gamma {i}"), MergeStrategy.REJECT)
    # Very high threshold returns nothing.
    res = await mp.query(
        MemoryQuery(
            task_instruction="alpha beta gamma",
            min_relevance=0.99,
            max_items=10,
            domain=None,
            session_id=None,
        )
    )
    assert res == []
    # max_items cap.
    res = await mp.query(
        MemoryQuery(
            task_instruction="alpha beta gamma",
            min_relevance=0.0,
            max_items=2,
            domain=None,
            session_id=None,
        )
    )
    assert len(res) == 2


@pytest.mark.asyncio
async def test_query_filters_by_domain() -> None:
    mp = StandardMemoryProvider()
    a = sem("a", "shared content").model_copy(update={"domain": "rust"})
    b = sem("b", "shared content").model_copy(update={"domain": "python"})
    await mp.store_semantic(a, MergeStrategy.REJECT)
    await mp.store_semantic(b, MergeStrategy.REJECT)
    res = await mp.query(
        MemoryQuery(
            task_instruction="shared content",
            domain="rust",
            min_relevance=0.0,
            max_items=10,
            session_id=None,
        )
    )
    assert len(res) == 1
    assert res[0].memory.id == MemoryId("a")


# ── Rule: Lifecycle — deprecate ────────────────────────────────────────


@pytest.mark.asyncio
async def test_deprecate_sets_status() -> None:
    mp = StandardMemoryProvider()
    await mp.store_semantic(sem("g1", "x"), MergeStrategy.REJECT)
    await mp.deprecate(MemoryId("g1"), "no longer needed")
    m = await mp.get_semantic(MemoryId("g1"))
    assert isinstance(m.status, MemoryStatusDeprecated)
    assert m.status.reason == "no longer needed"


@pytest.mark.asyncio
async def test_deprecate_unknown_id_not_found() -> None:
    mp = StandardMemoryProvider()
    with pytest.raises(NotFound):
        await mp.deprecate(MemoryId("nope"), "r")


# ── Rule: mark_pending_review ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_mark_pending_review_changes_status() -> None:
    mp = StandardMemoryProvider()
    await mp.store_semantic(sem("g1", "x"), MergeStrategy.REJECT)
    await mp.mark_pending_review(MemoryId("g1"))
    m = await mp.get_semantic(MemoryId("g1"))
    assert isinstance(m.status, MemoryStatusPendingReview)


# ── Rule: NotFound errors ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_semantic_unknown_returns_not_found() -> None:
    mp = StandardMemoryProvider()
    with pytest.raises(NotFound):
        await mp.get_semantic(MemoryId("nope"))


@pytest.mark.asyncio
async def test_get_episodic_unknown_session_returns_empty() -> None:
    mp = StandardMemoryProvider()
    res = await mp.get_episodic(SessionId("none"))
    assert res == []


# ── Episodic preserves insertion order across multiple writes ───────────


@pytest.mark.asyncio
async def test_episodic_preserves_order() -> None:
    mp = StandardMemoryProvider()
    for i in range(5):
        await mp.store_episodic(epi(f"e{i}", "s1", f"event {i}"))
    eps = await mp.get_episodic(SessionId("s1"))
    assert len(eps) == 5
    for i, e in enumerate(eps):
        assert e.id == MemoryId(f"e{i}")
