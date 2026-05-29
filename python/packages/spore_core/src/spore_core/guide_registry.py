"""GuideRegistry — manage lifecycle of guides/skills (issue #9).

Mirrors the Rust reference at
``rust/crates/spore-core/src/guide_registry.rs``.

Guides are feedforward artifacts injected before the agent acts: system
prompt fragments, skills loaded on demand, convention docs (AGENTS.md /
CLAUDE.md), schema annotations, and safety rules. This module is the
single source of truth for what guides exist, what state each is in, and
how each has performed across sessions.

See ``docs/harness-engineering-concepts.md`` § "GuideRegistry" for the
rules this module enforces.

Rules enforced:

* States ``Active`` / ``PendingReview`` / ``Deprecated`` / ``Stale``; no
  hard delete.
* :meth:`GuideRegistry.register` validates content (non-empty) and runs
  :meth:`GuideRegistry.check_conflicts` against existing ``Active``
  guides; conflicts surface as :class:`GuideConflictDetected`.
* :class:`GuideSourceMetaAgentProposed` forces ``PendingReview {
  AutomatedProposal, since=proposed_at }`` regardless of caller status.
* :meth:`GuideRegistry.select` returns only ``Active`` guides, filtered
  by domain and ``guide_types``, ordered by Jaccard relevance.
* :meth:`GuideRegistry.record_usage` appends an immutable history record
  and updates ``last_used``.
* :meth:`GuideRegistry.analyze_performance` emits
  :class:`ImprovementSignalGuideDeprecationRecommended`,
  :class:`ImprovementSignalSkillGenerationNeeded`, and
  :class:`ImprovementSignalConflictResolutionNeeded`.
* :meth:`GuideRegistry.promote_to_active` is the only path from
  ``PendingReview`` to ``Active``.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
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
from .memory import Timestamp
from .tool_registry import TaskPhase

# ============================================================================
# Identity
# ============================================================================

GuideId = NewType("GuideId", str)
"""Stable identifier for a registered guide."""


def new_guide_id(s: str) -> GuideId:
    return GuideId(s)


# ============================================================================
# Pydantic base
# ============================================================================


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ============================================================================
# GuideType
# ============================================================================


class GuideType(str, Enum):
    SYSTEM_PROMPT_FRAGMENT = "system_prompt_fragment"
    SKILL = "skill"
    CONVENTION_DOC = "convention_doc"
    SCHEMA_ANNOTATION = "schema_annotation"
    SAFETY_RULE = "safety_rule"


# ============================================================================
# GuideSource (discriminated union on ``kind``)
# ============================================================================


class GuideSourceManual(_Model):
    kind: Literal["manual"] = "manual"


class GuideSourceSessionGenerated(_Model):
    kind: Literal["session_generated"] = "session_generated"
    session_id: SessionId


class GuideSourceTraceDistilled(_Model):
    kind: Literal["trace_distilled"] = "trace_distilled"
    session_ids: list[SessionId] = Field(default_factory=list)


class GuideSourceMetaAgentProposed(_Model):
    kind: Literal["meta_agent_proposed"] = "meta_agent_proposed"
    proposed_at: Timestamp


GuideSource = Annotated[
    GuideSourceManual
    | GuideSourceSessionGenerated
    | GuideSourceTraceDistilled
    | GuideSourceMetaAgentProposed,
    Field(discriminator="kind"),
]


# ============================================================================
# PendingReason (discriminated union on ``kind``)
# ============================================================================


class PendingReasonAutomatedProposal(_Model):
    kind: Literal["automated_proposal"] = "automated_proposal"


class PendingReasonPerformanceDegradation(_Model):
    kind: Literal["performance_degradation"] = "performance_degradation"
    failure_rate_delta: float


class PendingReasonConflictDetected(_Model):
    kind: Literal["conflict_detected"] = "conflict_detected"
    conflicts_with: list[GuideId] = Field(default_factory=list)


class PendingReasonManualFlag(_Model):
    kind: Literal["manual_flag"] = "manual_flag"
    note: str


PendingReason = Annotated[
    PendingReasonAutomatedProposal
    | PendingReasonPerformanceDegradation
    | PendingReasonConflictDetected
    | PendingReasonManualFlag,
    Field(discriminator="kind"),
]


# ============================================================================
# GuideStatus (discriminated union on ``kind``)
# ============================================================================


class GuideStatusActive(_Model):
    kind: Literal["active"] = "active"


class GuideStatusPendingReview(_Model):
    kind: Literal["pending_review"] = "pending_review"
    reason: PendingReason
    since: Timestamp


class GuideStatusDeprecated(_Model):
    kind: Literal["deprecated"] = "deprecated"
    reason: str
    at: Timestamp


class GuideStatusStale(_Model):
    kind: Literal["stale"] = "stale"
    last_used: Timestamp


GuideStatus = Annotated[
    GuideStatusActive | GuideStatusPendingReview | GuideStatusDeprecated | GuideStatusStale,
    Field(discriminator="kind"),
]


# ============================================================================
# SessionOutcome (discriminated union on ``kind``)
# ============================================================================


class SessionOutcomeSuccess(_Model):
    kind: Literal["success"] = "success"


class SessionOutcomeFailure(_Model):
    kind: Literal["failure"] = "failure"
    reason: str


class SessionOutcomePartial(_Model):
    kind: Literal["partial"] = "partial"


class SessionOutcomeEscalated(_Model):
    """The session terminated cleanly because a tool escalated a structural
    signal to the harness's caller (issue #80, Tool Escalation Protocol).
    Distinct from ``Partial`` — an escalation is an intentional, clean terminal
    outcome, not a partial success."""

    kind: Literal["escalated"] = "escalated"


SessionOutcome = Annotated[
    SessionOutcomeSuccess | SessionOutcomeFailure | SessionOutcomePartial | SessionOutcomeEscalated,
    Field(discriminator="kind"),
]


# ============================================================================
# Records
# ============================================================================


class Guide(_Model):
    id: GuideId
    name: str
    content: str
    guide_type: GuideType
    domain: str | None = None
    source: GuideSource
    status: GuideStatus
    created_at: Timestamp
    last_used: Timestamp | None = None
    version: int = 1


class GuideUsageRecord(_Model):
    guide_id: GuideId
    session_id: SessionId
    task_domain: str | None = None
    outcome: SessionOutcome
    recorded_at: Timestamp


class GuideQuery(_Model):
    task_instruction: str
    domain: str | None = None
    phase: TaskPhase | None = None
    guide_types: list[GuideType] = Field(default_factory=list)


class GuideConflict(_Model):
    guide_a: GuideId
    guide_b: GuideId
    reason: str


# ============================================================================
# ImprovementSignal (discriminated union on ``kind``)
# ============================================================================


class ImprovementSignalSkillGenerationNeeded(_Model):
    kind: Literal["skill_generation_needed"] = "skill_generation_needed"
    pattern: str
    session_ids: list[SessionId] = Field(default_factory=list)


class ImprovementSignalGuideDeprecationRecommended(_Model):
    kind: Literal["guide_deprecation_recommended"] = "guide_deprecation_recommended"
    guide_id: GuideId
    reason: str


class ImprovementSignalConflictResolutionNeeded(_Model):
    kind: Literal["conflict_resolution_needed"] = "conflict_resolution_needed"
    conflict: GuideConflict


ImprovementSignal = Annotated[
    ImprovementSignalSkillGenerationNeeded
    | ImprovementSignalGuideDeprecationRecommended
    | ImprovementSignalConflictResolutionNeeded,
    Field(discriminator="kind"),
]


# ============================================================================
# Errors
# ============================================================================


class GuideRegistryError(SporeError):
    """Root of every error raised by a :class:`GuideRegistry`."""

    kind: ClassVar[str] = "GuideRegistryError"


class GuideNotFound(GuideRegistryError):
    kind: ClassVar[str] = "NotFound"

    def __init__(self, id: GuideId) -> None:
        self.id = id
        super().__init__(f"guide not found: {id!r}")


class GuideConflictDetected(GuideRegistryError):
    kind: ClassVar[str] = "ConflictDetected"

    def __init__(self, conflict: GuideConflict) -> None:
        self.conflict = conflict
        super().__init__(
            f"conflict detected between {conflict.guide_a!r} and {conflict.guide_b!r}: "
            f"{conflict.reason}"
        )


class GuideValidationFailed(GuideRegistryError):
    kind: ClassVar[str] = "ValidationFailed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"validation failed: {reason}")


class GuideStorageError(GuideRegistryError):
    kind: ClassVar[str] = "StorageError"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"storage error: {reason}")


# ============================================================================
# Protocol
# ============================================================================


@runtime_checkable
class GuideRegistry(Protocol):
    """Canonical issue-#9 ``GuideRegistry`` interface."""

    async def register(self, guide: Guide) -> GuideId: ...

    async def select(self, query: GuideQuery) -> list[Guide]: ...

    async def record_usage(self, record: GuideUsageRecord) -> None: ...

    async def usage_history(self, id: GuideId) -> list[GuideUsageRecord]: ...

    async def deprecate(self, id: GuideId, reason: str) -> None: ...

    async def mark_pending_review(self, id: GuideId, reason: PendingReason) -> None: ...

    async def promote_to_active(self, id: GuideId) -> None: ...

    async def analyze_performance(
        self,
        window: timedelta,
        min_failure_rate_delta: float,
        min_pattern_occurrences: int,
    ) -> list[ImprovementSignal]: ...

    async def check_conflicts(self, content: str, domain: str | None) -> list[GuideConflict]: ...


# ============================================================================
# Helpers
# ============================================================================


_TOKEN_RE = re.compile(r"[a-z0-9]+")
CONFLICT_THRESHOLD: float = 0.6


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
        raise GuideValidationFailed("content must not be empty")


def _enforce_meta_agent_pending(guide: Guide) -> Guide:
    if isinstance(guide.source, GuideSourceMetaAgentProposed):
        return guide.model_copy(
            update={
                "status": GuideStatusPendingReview(
                    reason=PendingReasonAutomatedProposal(),
                    since=guide.source.proposed_at,
                ),
            }
        )
    return guide


def _parse_rfc3339(s: str) -> datetime | None:
    """Parse an RFC 3339 timestamp; tolerates trailing ``Z``."""
    if not s:
        return None
    try:
        text = s.replace("Z", "+00:00") if s.endswith("Z") else s
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _format_rfc3339(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _system_now() -> Timestamp:
    return Timestamp(_format_rfc3339(datetime.now(tz=timezone.utc)))


def _in_window(ts: Timestamp, cutoff: Timestamp | None) -> bool:
    if cutoff is None:
        return True
    # Lexicographic RFC 3339 compare — ISO 8601 UTC sorts correctly.
    return str(ts) >= str(cutoff)


def _compute_cutoff(now: Timestamp, window: timedelta) -> Timestamp | None:
    dt = _parse_rfc3339(str(now))
    if dt is None:
        return None
    cutoff_dt = dt - timedelta(seconds=window.total_seconds())
    return Timestamp(_format_rfc3339(cutoff_dt))


# ============================================================================
# StandardGuideRegistry — reference in-memory implementation
# ============================================================================


class StandardGuideRegistry:
    """Reference :class:`GuideRegistry`. In-memory; suitable for tests and
    short-lived processes.

    Conflict heuristic: two guides conflict when they share a domain and
    have Jaccard token overlap >= :data:`CONFLICT_THRESHOLD` but
    non-identical content. Production deployments override
    :meth:`check_conflicts`.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._guides: dict[GuideId, Guide] = {}
        self._usage: list[GuideUsageRecord] = []
        self._now_override: Timestamp | None = None

    # ── Test helper ────────────────────────────────────────────────────────

    def set_now(self, ts: Timestamp) -> None:
        """Pin the "now" timestamp for window math. Tests use this to keep
        :meth:`analyze_performance` deterministic without a real clock."""
        self._now_override = ts

    def _now(self) -> Timestamp:
        return self._now_override if self._now_override is not None else _system_now()

    # ── register ───────────────────────────────────────────────────────────

    async def register(self, guide: Guide) -> GuideId:
        _validate_content(guide.content)
        guide = _enforce_meta_agent_pending(guide)

        # Conflict check against existing Active guides.
        conflicts = await self.check_conflicts(guide.content, guide.domain)
        if conflicts:
            first = conflicts[0]
            rewritten = GuideConflict(
                guide_a=guide.id,
                guide_b=first.guide_b,
                reason=first.reason,
            )
            raise GuideConflictDetected(rewritten)

        async with self._lock:
            self._guides[guide.id] = guide
            return guide.id

    # ── select ─────────────────────────────────────────────────────────────

    async def select(self, query: GuideQuery) -> list[Guide]:
        async with self._lock:
            candidates = list(self._guides.values())

        scored: list[tuple[float, Guide]] = []
        for g in candidates:
            if not isinstance(g.status, GuideStatusActive):
                continue
            # Domain filter: ``Some(d)`` only matches ``Some(d)``;
            # ``Some(d)`` skips ``None``; ``None`` matches anything.
            if query.domain is not None:
                if g.domain is None or g.domain != query.domain:
                    continue
            if query.guide_types and g.guide_type not in query.guide_types:
                continue
            scored.append((_jaccard(query.task_instruction, g.content), g))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [g for _, g in scored]

    # ── record_usage ───────────────────────────────────────────────────────

    async def record_usage(self, record: GuideUsageRecord) -> None:
        async with self._lock:
            g = self._guides.get(record.guide_id)
            if g is None:
                raise GuideNotFound(record.guide_id)
            self._guides[record.guide_id] = g.model_copy(update={"last_used": record.recorded_at})
            self._usage.append(record)

    # ── usage_history ──────────────────────────────────────────────────────

    async def usage_history(self, id: GuideId) -> list[GuideUsageRecord]:
        async with self._lock:
            if id not in self._guides:
                raise GuideNotFound(id)
            return [r for r in self._usage if r.guide_id == id]

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def deprecate(self, id: GuideId, reason: str) -> None:
        async with self._lock:
            g = self._guides.get(id)
            if g is None:
                raise GuideNotFound(id)
            self._guides[id] = g.model_copy(
                update={
                    "status": GuideStatusDeprecated(reason=reason, at=self._now()),
                }
            )

    async def mark_pending_review(self, id: GuideId, reason: PendingReason) -> None:
        async with self._lock:
            g = self._guides.get(id)
            if g is None:
                raise GuideNotFound(id)
            self._guides[id] = g.model_copy(
                update={
                    "status": GuideStatusPendingReview(reason=reason, since=self._now()),
                }
            )

    async def promote_to_active(self, id: GuideId) -> None:
        async with self._lock:
            g = self._guides.get(id)
            if g is None:
                raise GuideNotFound(id)
            if not isinstance(g.status, GuideStatusPendingReview):
                raise GuideValidationFailed("promote_to_active requires PendingReview status")
            self._guides[id] = g.model_copy(update={"status": GuideStatusActive()})

    # ── analyze_performance ────────────────────────────────────────────────

    async def analyze_performance(
        self,
        window: timedelta,
        min_failure_rate_delta: float,
        min_pattern_occurrences: int,
    ) -> list[ImprovementSignal]:
        async with self._lock:
            now = self._now()
            guides = dict(self._guides)
            usage = list(self._usage)

        cutoff = _compute_cutoff(now, window)
        in_win = [r for r in usage if _in_window(r.recorded_at, cutoff)]

        signals: list[ImprovementSignal] = []

        total_failures = sum(1 for r in in_win if isinstance(r.outcome, SessionOutcomeFailure))
        total_records = len(in_win)

        for gid, guide in guides.items():
            # Conflict-derived signal.
            if isinstance(guide.status, GuideStatusPendingReview) and isinstance(
                guide.status.reason, PendingReasonConflictDetected
            ):
                conflicts_with = guide.status.reason.conflicts_with
                if conflicts_with:
                    other = conflicts_with[0]
                    signals.append(
                        ImprovementSignalConflictResolutionNeeded(
                            conflict=GuideConflict(
                                guide_a=gid,
                                guide_b=other,
                                reason="pending-review conflict",
                            )
                        )
                    )

            with_records = [r for r in in_win if r.guide_id == gid]
            if not with_records:
                continue
            with_fail = sum(1 for r in with_records if isinstance(r.outcome, SessionOutcomeFailure))
            with_rate = with_fail / len(with_records)
            without_count = total_records - len(with_records)
            if without_count <= 0:
                baseline = 0.0
            else:
                baseline = (total_failures - with_fail) / without_count
            if with_rate - baseline >= min_failure_rate_delta:
                signals.append(
                    ImprovementSignalGuideDeprecationRecommended(
                        guide_id=gid,
                        reason=(
                            f"failure-rate delta {with_rate - baseline:.2f} "
                            f"(with={with_rate:.2f} vs baseline={baseline:.2f})"
                        ),
                    )
                )

        # Pattern detection: failure reasons appearing >= N times.
        pattern_counts: dict[str, list[SessionId]] = defaultdict(list)
        for r in in_win:
            if isinstance(r.outcome, SessionOutcomeFailure):
                pattern_counts[r.outcome.reason].append(r.session_id)
        for pattern, sessions in pattern_counts.items():
            if len(sessions) >= min_pattern_occurrences:
                signals.append(
                    ImprovementSignalSkillGenerationNeeded(
                        pattern=pattern,
                        session_ids=sessions,
                    )
                )

        return signals

    # ── check_conflicts ────────────────────────────────────────────────────

    async def check_conflicts(self, content: str, domain: str | None) -> list[GuideConflict]:
        async with self._lock:
            guides = list(self._guides.values())

        out: list[GuideConflict] = []
        for g in guides:
            if not isinstance(g.status, GuideStatusActive):
                continue
            # Same-domain only (None vs Some is not a conflict).
            if g.domain != domain:
                continue
            if g.content == content:
                continue
            score = _jaccard(g.content, content)
            if score >= CONFLICT_THRESHOLD:
                out.append(
                    GuideConflict(
                        # Placeholder; ``register`` rewrites this with the new id.
                        guide_a=GuideId("<new>"),
                        guide_b=g.id,
                        reason=f"Jaccard overlap {score:.2f} >= {CONFLICT_THRESHOLD}",
                    )
                )
        return out


__all__ = [
    "CONFLICT_THRESHOLD",
    "Guide",
    "GuideConflict",
    "GuideConflictDetected",
    "GuideId",
    "GuideNotFound",
    "GuideQuery",
    "GuideRegistry",
    "GuideRegistryError",
    "GuideSource",
    "GuideSourceManual",
    "GuideSourceMetaAgentProposed",
    "GuideSourceSessionGenerated",
    "GuideSourceTraceDistilled",
    "GuideStatus",
    "GuideStatusActive",
    "GuideStatusDeprecated",
    "GuideStatusPendingReview",
    "GuideStatusStale",
    "GuideStorageError",
    "GuideType",
    "GuideUsageRecord",
    "GuideValidationFailed",
    "ImprovementSignal",
    "ImprovementSignalConflictResolutionNeeded",
    "ImprovementSignalGuideDeprecationRecommended",
    "ImprovementSignalSkillGenerationNeeded",
    "PendingReason",
    "PendingReasonAutomatedProposal",
    "PendingReasonConflictDetected",
    "PendingReasonManualFlag",
    "PendingReasonPerformanceDegradation",
    "SessionOutcome",
    "SessionOutcomeEscalated",
    "SessionOutcomeFailure",
    "SessionOutcomePartial",
    "SessionOutcomeSuccess",
    "StandardGuideRegistry",
    "new_guide_id",
]
