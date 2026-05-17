"""Tests for :class:`StandardGuideRegistry` — issue #9.

Mirrors the Rust unit tests in
``rust/crates/spore-core/src/guide_registry.rs`` ``tests`` module.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from spore_core.guide_registry import (
    Guide,
    GuideConflictDetected,
    GuideId,
    GuideNotFound,
    GuideQuery,
    GuideSourceManual,
    GuideSourceMetaAgentProposed,
    GuideStatusActive,
    GuideStatusDeprecated,
    GuideStatusPendingReview,
    GuideType,
    GuideUsageRecord,
    GuideValidationFailed,
    ImprovementSignalGuideDeprecationRecommended,
    ImprovementSignalSkillGenerationNeeded,
    PendingReasonAutomatedProposal,
    PendingReasonManualFlag,
    SessionOutcomeFailure,
    SessionOutcomeSuccess,
    StandardGuideRegistry,
    _compute_cutoff,
    _parse_rfc3339,
)
from spore_core.harness import SessionId
from spore_core.memory import Timestamp


def ts(s: str) -> Timestamp:
    return Timestamp(s)


def make_guide(id_: str, content: str) -> Guide:
    return Guide(
        id=GuideId(id_),
        name=id_,
        content=content,
        guide_type=GuideType.SKILL,
        domain=None,
        source=GuideSourceManual(),
        status=GuideStatusActive(),
        created_at=ts("2026-05-16T00:00:00Z"),
        last_used=None,
        version=1,
    )


def usage(gid: str, sid: str, outcome) -> GuideUsageRecord:
    return GuideUsageRecord(
        guide_id=GuideId(gid),
        session_id=SessionId(sid),
        task_domain=None,
        outcome=outcome,
        recorded_at=ts("2026-05-16T00:00:00Z"),
    )


# ── Rule: register validates content ────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_empty_content_fails() -> None:
    r = StandardGuideRegistry()
    g = make_guide("g1", "   ")
    with pytest.raises(GuideValidationFailed):
        await r.register(g)


# ── Rule: MetaAgentProposed forces PendingReview ────────────────────────────


@pytest.mark.asyncio
async def test_meta_agent_source_forces_pending_review() -> None:
    r = StandardGuideRegistry()
    g = make_guide("g1", "proposed")
    g = g.model_copy(
        update={
            "source": GuideSourceMetaAgentProposed(proposed_at=ts("2026-05-16T01:00:00Z")),
            "status": GuideStatusActive(),  # caller lies; provider corrects
        }
    )
    await r.register(g)
    sel = await r.select(GuideQuery(task_instruction="anything"))
    # Not Active → not in select.
    assert all(s.id != "g1" for s in sel)
    stored = r._guides[GuideId("g1")]
    assert isinstance(stored.status, GuideStatusPendingReview)
    assert isinstance(stored.status.reason, PendingReasonAutomatedProposal)


# ── Rule: select filters by status, domain, and type ────────────────────────


@pytest.mark.asyncio
async def test_select_filters_by_status_domain_and_type() -> None:
    r = StandardGuideRegistry()
    a = make_guide("a", "rust async tokio runtime").model_copy(update={"domain": "rust"})
    b = make_guide("b", "pytest fixtures python").model_copy(
        update={"domain": "python", "guide_type": GuideType.CONVENTION_DOC}
    )
    c = make_guide("c", "deprecated content").model_copy(
        update={
            "status": GuideStatusDeprecated(reason="old", at=ts("2026-05-16T00:00:00Z")),
        }
    )
    await r.register(a)
    await r.register(b)
    # Inject c directly (bypasses register validation).
    r._guides[GuideId("c")] = c

    res = await r.select(
        GuideQuery(
            task_instruction="rust tokio",
            domain="rust",
            phase=None,
            guide_types=[GuideType.SKILL],
        )
    )
    assert len(res) == 1
    assert res[0].id == "a"


# ── Rule: select sorted by relevance ────────────────────────────────────────


@pytest.mark.asyncio
async def test_select_sorted_by_relevance() -> None:
    r = StandardGuideRegistry()
    await r.register(make_guide("a", "alpha beta gamma delta"))
    await r.register(make_guide("b", "zebra"))
    await r.register(make_guide("c", "alpha beta"))
    res = await r.select(GuideQuery(task_instruction="alpha beta"))
    assert res[0].id == "c"
    assert res[-1].id == "b"


# ── Rule: conflict detection at registration ────────────────────────────────


@pytest.mark.asyncio
async def test_register_detects_conflict_in_same_domain() -> None:
    r = StandardGuideRegistry()
    existing = make_guide("a", "always run tests before commit").model_copy(
        update={"domain": "rust"}
    )
    await r.register(existing)

    conflicting = make_guide("b", "always run tests before committing").model_copy(
        update={"domain": "rust"}
    )
    with pytest.raises(GuideConflictDetected) as ei:
        await r.register(conflicting)
    assert ei.value.conflict.guide_a == "b"
    assert ei.value.conflict.guide_b == "a"


@pytest.mark.asyncio
async def test_no_conflict_across_domains() -> None:
    r = StandardGuideRegistry()
    a = make_guide("a", "always run tests before commit").model_copy(update={"domain": "rust"})
    await r.register(a)
    b = make_guide("b", "always run tests before commit").model_copy(update={"domain": "python"})
    await r.register(b)


# ── Rule: record_usage requires existing guide & updates last_used ──────────


@pytest.mark.asyncio
async def test_record_usage_requires_known_guide() -> None:
    r = StandardGuideRegistry()
    with pytest.raises(GuideNotFound):
        await r.record_usage(usage("nope", "s1", SessionOutcomeSuccess()))


@pytest.mark.asyncio
async def test_record_usage_updates_last_used() -> None:
    r = StandardGuideRegistry()
    await r.register(make_guide("a", "x"))
    u = usage("a", "s1", SessionOutcomeSuccess()).model_copy(
        update={"recorded_at": ts("2026-06-01T00:00:00Z")}
    )
    await r.record_usage(u)
    hist = await r.usage_history(GuideId("a"))
    assert len(hist) == 1
    assert r._guides[GuideId("a")].last_used == "2026-06-01T00:00:00Z"


# ── Rule: deprecate sets status ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deprecate_sets_status_and_404s_on_missing() -> None:
    r = StandardGuideRegistry()
    await r.register(make_guide("a", "x"))
    await r.deprecate(GuideId("a"), "obsolete")
    stored = r._guides[GuideId("a")]
    assert isinstance(stored.status, GuideStatusDeprecated)
    assert stored.status.reason == "obsolete"
    with pytest.raises(GuideNotFound):
        await r.deprecate(GuideId("nope"), "x")


# ── Rule: promote_to_active only from PendingReview ─────────────────────────


@pytest.mark.asyncio
async def test_promote_to_active_only_from_pending_review() -> None:
    r = StandardGuideRegistry()
    await r.register(make_guide("a", "x"))
    with pytest.raises(GuideValidationFailed):
        await r.promote_to_active(GuideId("a"))
    await r.mark_pending_review(GuideId("a"), PendingReasonManualFlag(note="x"))
    await r.promote_to_active(GuideId("a"))
    assert isinstance(r._guides[GuideId("a")].status, GuideStatusActive)


# ── Rule: analyze_performance flags high-delta failure rate ─────────────────


@pytest.mark.asyncio
async def test_analyze_performance_flags_high_failure_rate() -> None:
    r = StandardGuideRegistry()
    r.set_now(ts("2026-05-16T01:00:00Z"))
    await r.register(make_guide("a", "x"))
    await r.register(make_guide("b", "y"))
    for i in range(3):
        await r.record_usage(usage("a", f"s{i}", SessionOutcomeFailure(reason="boom")))
    for i in range(3):
        await r.record_usage(usage("b", f"sb{i}", SessionOutcomeSuccess()))
    signals = await r.analyze_performance(timedelta(days=1), 0.5, 100)
    dep_for_a = any(
        isinstance(s, ImprovementSignalGuideDeprecationRecommended) and s.guide_id == "a"
        for s in signals
    )
    assert dep_for_a


# ── Rule: SkillGenerationNeeded on repeated pattern ─────────────────────────


@pytest.mark.asyncio
async def test_analyze_performance_emits_skill_generation_for_repeated_pattern() -> None:
    r = StandardGuideRegistry()
    r.set_now(ts("2026-05-16T01:00:00Z"))
    await r.register(make_guide("a", "x"))
    for i in range(4):
        await r.record_usage(
            usage("a", f"s{i}", SessionOutcomeFailure(reason="panic: index out of bounds"))
        )
    signals = await r.analyze_performance(timedelta(days=1), 999.0, 3)
    gen = any(
        isinstance(s, ImprovementSignalSkillGenerationNeeded)
        and s.pattern == "panic: index out of bounds"
        and len(s.session_ids) == 4
        for s in signals
    )
    assert gen


# ── Rule: analyze_performance honors window ─────────────────────────────────


@pytest.mark.asyncio
async def test_analyze_performance_filters_by_window() -> None:
    r = StandardGuideRegistry()
    r.set_now(ts("2026-05-16T00:00:00Z"))
    await r.register(make_guide("a", "x"))
    old = usage("a", "s0", SessionOutcomeFailure(reason="old-pattern")).model_copy(
        update={"recorded_at": ts("2020-01-01T00:00:00Z")}
    )
    await r.record_usage(old)
    signals = await r.analyze_performance(timedelta(seconds=3600), 0.0, 1)
    any_pattern = any(isinstance(s, ImprovementSignalSkillGenerationNeeded) for s in signals)
    assert not any_pattern


# ── Rule: usage_history filters to one guide ────────────────────────────────


@pytest.mark.asyncio
async def test_usage_history_filters_to_one_guide() -> None:
    r = StandardGuideRegistry()
    await r.register(make_guide("a", "x"))
    await r.register(make_guide("b", "y"))
    await r.record_usage(usage("a", "s1", SessionOutcomeSuccess()))
    await r.record_usage(usage("b", "s2", SessionOutcomeSuccess()))
    h = await r.usage_history(GuideId("a"))
    assert len(h) == 1
    assert h[0].guide_id == "a"


# ── Rule: check_conflicts external API ──────────────────────────────────────


@pytest.mark.asyncio
async def test_check_conflicts_does_not_flag_identical_content() -> None:
    r = StandardGuideRegistry()
    await r.register(make_guide("a", "same exact content"))
    conflicts = await r.check_conflicts("same exact content", None)
    assert conflicts == []


# ── Rule: select returns empty when no guides ───────────────────────────────


@pytest.mark.asyncio
async def test_select_empty_when_no_guides() -> None:
    r = StandardGuideRegistry()
    res = await r.select(GuideQuery(task_instruction="anything"))
    assert res == []


# ── Date arithmetic sanity ──────────────────────────────────────────────────


def test_rfc3339_round_trip() -> None:
    dt = _parse_rfc3339("2026-05-16T12:34:56Z")
    assert dt is not None
    cutoff = _compute_cutoff(Timestamp("2026-05-16T12:34:56Z"), timedelta(seconds=0))
    assert cutoff == "2026-05-16T12:34:56Z"
