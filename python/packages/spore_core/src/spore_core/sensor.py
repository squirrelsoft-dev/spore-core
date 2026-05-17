"""SensorChain — post-action feedback controls and output quality evaluation
(issue #10).

Mirrors the Rust reference at ``rust/crates/spore-core/src/sensor.rs``.

Sensors observe the agent's actions (tool calls, tool results, agent
responses) at defined trigger points and emit :class:`SensorResult` values.
The chain is a registry plus a fan-out evaluator: it runs every sensor
registered for a trigger and returns all results without short-circuiting.
The harness decides routing (Warn → inject observation; Halt → stop).

See ``docs/harness-engineering-concepts.md`` § "SensorChain" for the rules
this module enforces.

Rules enforced:

* :meth:`SensorChain.fire` runs every sensor whose ``config.triggers``
  contains the trigger and returns all results — the chain never
  short-circuits.
* :class:`SensorKind.COMPUTATIONAL` sensors run on every matching trigger.
* :class:`SensorKind.INFERENTIAL` sensors are gated by
  ``run_every_n_turns`` (modulo the :attr:`SensorInput.turn_number`) and
  ``run_on_phases`` (if set, the input's ``phase`` must match).
* :meth:`SensorChain.stats` aggregates fire history; ``fire_rate`` is
  ``total_fires / sessions_seen`` clamped to ``[0.0, 1.0]``.
* :meth:`SensorChain.signal_quality_report` flags
  :class:`SensorSignalFlagNeverFired` and
  :class:`SensorSignalFlagAlwaysFiring`.
* Trigger matching for :class:`SensorTriggerPostTool`: an empty
  ``tool_name`` matches any tool; a non-empty ``tool_name`` matches
  exact equality only.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from enum import Enum
from typing import Annotated, ClassVar, Literal, NewType, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .errors import SporeError
from .harness import SessionId, SessionState
from .memory import Timestamp
from .model import ToolCall, ToolResult
from .tool_registry import TaskPhase

# ============================================================================
# Identity
# ============================================================================

SensorId = NewType("SensorId", str)
"""Stable identifier for a registered sensor."""


def new_sensor_id(s: str) -> SensorId:
    return SensorId(s)


# ============================================================================
# Pydantic base
# ============================================================================


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ============================================================================
# Enums
# ============================================================================


class SensorKind(str, Enum):
    COMPUTATIONAL = "computational"
    INFERENTIAL = "inferential"


class SensorOutcome(str, Enum):
    PASS = "pass"
    WARN = "warn"
    HALT = "halt"


# ============================================================================
# SensorTrigger (discriminated union on ``kind``)
# ============================================================================


class SensorTriggerPostTool(_Model):
    kind: Literal["post_tool"] = "post_tool"
    tool_name: str = ""


class SensorTriggerPostTurn(_Model):
    kind: Literal["post_turn"] = "post_turn"


class SensorTriggerPostSession(_Model):
    kind: Literal["post_session"] = "post_session"


class SensorTriggerContinuous(_Model):
    kind: Literal["continuous"] = "continuous"


class SensorTriggerOnToolError(_Model):
    kind: Literal["on_tool_error"] = "on_tool_error"


class SensorTriggerOnCompaction(_Model):
    kind: Literal["on_compaction"] = "on_compaction"


SensorTrigger = Annotated[
    SensorTriggerPostTool
    | SensorTriggerPostTurn
    | SensorTriggerPostSession
    | SensorTriggerContinuous
    | SensorTriggerOnToolError
    | SensorTriggerOnCompaction,
    Field(discriminator="kind"),
]


def _trigger_matches(configured: object, fired: object) -> bool:
    """True if a configured trigger matches the trigger that fired.

    ``SensorTriggerPostTool(tool_name="")`` is a wildcard matching any
    ``post_tool`` trigger.
    """
    if isinstance(configured, SensorTriggerPostTool) and isinstance(fired, SensorTriggerPostTool):
        return configured.tool_name == "" or configured.tool_name == fired.tool_name
    return type(configured) is type(fired)


# ============================================================================
# Records
# ============================================================================


class SensorInput(_Model):
    session_id: SessionId
    turn_number: int | None = None
    phase: TaskPhase | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    agent_response: str | None = None
    session_state: SessionState = Field(default_factory=SessionState)


class SensorResult(_Model):
    sensor_id: SensorId
    outcome: SensorOutcome
    observation: str | None = None
    detail: str
    fired_at: Timestamp


class SensorSignalThresholds(_Model):
    never_fired_after_n_sessions: int = 10
    always_fired_rate: float = 0.9


class SensorConfig(_Model):
    id: SensorId
    name: str
    kind: SensorKind
    triggers: list[SensorTrigger]
    run_every_n_turns: int | None = None
    run_on_phases: list[TaskPhase] | None = None
    low_signal_threshold: SensorSignalThresholds = Field(default_factory=SensorSignalThresholds)


class SensorStats(_Model):
    sensor_id: SensorId
    total_fires: int
    warn_count: int
    halt_count: int
    pass_count: int
    fire_rate: float
    last_fired: Timestamp | None = None
    low_signal_flag: bool


# ============================================================================
# SensorSignalFlag (discriminated union on ``kind``)
# ============================================================================


class SensorSignalFlagNeverFired(_Model):
    kind: Literal["never_fired"] = "never_fired"
    sensor_id: SensorId
    sessions_observed: int


class SensorSignalFlagAlwaysFiring(_Model):
    kind: Literal["always_firing"] = "always_firing"
    sensor_id: SensorId
    fire_rate: float


SensorSignalFlag = Annotated[
    SensorSignalFlagNeverFired | SensorSignalFlagAlwaysFiring,
    Field(discriminator="kind"),
]


# ============================================================================
# Errors
# ============================================================================


class SensorError(SporeError):
    """Root of every error raised by a :class:`SensorChain`."""

    kind: ClassVar[str] = "SensorError"


class SensorAlreadyRegistered(SensorError):
    kind: ClassVar[str] = "AlreadyRegistered"

    def __init__(self, id: SensorId) -> None:
        self.id = id
        super().__init__(f"sensor already registered: {id!r}")


class SensorValidationFailed(SensorError):
    kind: ClassVar[str] = "ValidationFailed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"validation failed: {reason}")


# ============================================================================
# Protocols
# ============================================================================


@runtime_checkable
class Sensor(Protocol):
    """A single sensor."""

    async def evaluate(self, input: SensorInput) -> SensorResult: ...

    def config(self) -> SensorConfig: ...


@runtime_checkable
class SensorChain(Protocol):
    """Registry + fan-out evaluator."""

    async def register(self, sensor: Sensor) -> None: ...

    async def fire(self, trigger: SensorTrigger, input: SensorInput) -> list[SensorResult]: ...

    async def stats(self, since: Timestamp | None = None) -> list[SensorStats]: ...

    async def signal_quality_report(self, min_sessions: int) -> list[SensorSignalFlag]: ...


# ============================================================================
# Helpers
# ============================================================================


def _inferential_gate_open(cfg: SensorConfig, input: SensorInput) -> bool:
    """Decide whether an inferential sensor should run on this input given
    its gating fields. Computational sensors are unconditional."""
    if cfg.kind == SensorKind.COMPUTATIONAL:
        return True
    if cfg.run_on_phases is not None:
        if len(cfg.run_on_phases) > 0:
            if input.phase is None or input.phase not in cfg.run_on_phases:
                return False
    if cfg.run_every_n_turns is not None:
        n = cfg.run_every_n_turns
        if n == 0:
            return False
        # Fire when turn_number is a multiple of n. Missing turn_number
        # means we cannot gate — default to firing.
        if input.turn_number is not None:
            if input.turn_number % n != 0:
                return False
    return True


# ============================================================================
# StandardSensorChain — reference in-memory implementation
# ============================================================================


class _HistoryRecord:
    __slots__ = ("sensor_id", "session_id", "outcome", "fired_at")

    def __init__(
        self,
        sensor_id: SensorId,
        session_id: SessionId,
        outcome: SensorOutcome,
        fired_at: Timestamp,
    ) -> None:
        self.sensor_id = sensor_id
        self.session_id = session_id
        self.outcome = outcome
        self.fired_at = fired_at


class StandardSensorChain:
    """Reference :class:`SensorChain`. In-memory; suitable for tests and
    short-lived processes."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sensors: list[tuple[SensorConfig, Sensor]] = []
        self._history: list[_HistoryRecord] = []
        self._sessions_seen: set[SessionId] = set()

    # ── register ───────────────────────────────────────────────────────────

    async def register(self, sensor: Sensor) -> None:
        cfg = sensor.config()
        if not cfg.triggers:
            raise SensorValidationFailed("sensor must declare at least one trigger")
        async with self._lock:
            if any(c.id == cfg.id for c, _ in self._sensors):
                raise SensorAlreadyRegistered(cfg.id)
            self._sensors.append((cfg, sensor))

    # ── fire ───────────────────────────────────────────────────────────────

    async def fire(self, trigger: SensorTrigger, input: SensorInput) -> list[SensorResult]:
        async with self._lock:
            self._sessions_seen.add(input.session_id)
            candidates: list[tuple[SensorId, Sensor]] = [
                (cfg.id, sensor)
                for cfg, sensor in self._sensors
                if any(_trigger_matches(t, trigger) for t in cfg.triggers)
                and _inferential_gate_open(cfg, input)
            ]

        results: list[SensorResult] = []
        for sensor_id, sensor in candidates:
            result = await sensor.evaluate(input)
            async with self._lock:
                self._history.append(
                    _HistoryRecord(
                        sensor_id=sensor_id,
                        session_id=input.session_id,
                        outcome=result.outcome,
                        fired_at=result.fired_at,
                    )
                )
            results.append(result)
        return results

    # ── stats ──────────────────────────────────────────────────────────────

    async def stats(self, since: Timestamp | None = None) -> list[SensorStats]:
        async with self._lock:
            sensors = list(self._sensors)
            history = list(self._history)
            sessions_total = len(self._sessions_seen)

        agg_total: dict[SensorId, int] = defaultdict(int)
        agg_pass: dict[SensorId, int] = defaultdict(int)
        agg_warn: dict[SensorId, int] = defaultdict(int)
        agg_halt: dict[SensorId, int] = defaultdict(int)
        agg_last: dict[SensorId, Timestamp] = {}
        for cfg, _ in sensors:
            agg_total[cfg.id] = 0  # ensure presence

        for rec in history:
            if since is not None and str(rec.fired_at) < str(since):
                continue
            agg_total[rec.sensor_id] += 1
            if rec.outcome == SensorOutcome.PASS:
                agg_pass[rec.sensor_id] += 1
            elif rec.outcome == SensorOutcome.WARN:
                agg_warn[rec.sensor_id] += 1
            elif rec.outcome == SensorOutcome.HALT:
                agg_halt[rec.sensor_id] += 1
            agg_last[rec.sensor_id] = rec.fired_at

        out: list[SensorStats] = []
        cfg_by_id = {cfg.id: cfg for cfg, _ in sensors}
        sensor_ids: set[SensorId] = set(agg_total.keys()) | set(cfg_by_id.keys())
        for sid in sensor_ids:
            total = agg_total.get(sid, 0)
            if sessions_total == 0:
                fire_rate = 0.0
            else:
                fire_rate = max(0.0, min(1.0, total / sessions_total))
            cfg = cfg_by_id.get(sid)
            low_signal_flag = False
            if cfg is not None:
                low_signal_flag = fire_rate > cfg.low_signal_threshold.always_fired_rate or (
                    total == 0
                    and sessions_total >= cfg.low_signal_threshold.never_fired_after_n_sessions
                )
            out.append(
                SensorStats(
                    sensor_id=sid,
                    total_fires=total,
                    warn_count=agg_warn.get(sid, 0),
                    halt_count=agg_halt.get(sid, 0),
                    pass_count=agg_pass.get(sid, 0),
                    fire_rate=fire_rate,
                    last_fired=agg_last.get(sid),
                    low_signal_flag=low_signal_flag,
                )
            )
        out.sort(key=lambda s: s.sensor_id)
        return out

    # ── signal_quality_report ──────────────────────────────────────────────

    async def signal_quality_report(self, min_sessions: int) -> list[SensorSignalFlag]:
        async with self._lock:
            sensors = list(self._sensors)
            history = list(self._history)
            sessions_observed = len(self._sessions_seen)

        if sessions_observed < min_sessions:
            return []

        out: list[SensorSignalFlag] = []
        for cfg, _ in sensors:
            fires = [r for r in history if r.sensor_id == cfg.id]
            total = len(fires)
            if sessions_observed == 0:
                fire_rate = 0.0
            else:
                fire_rate = max(0.0, min(1.0, total / sessions_observed))

            if (
                total == 0
                and sessions_observed >= cfg.low_signal_threshold.never_fired_after_n_sessions
            ):
                out.append(
                    SensorSignalFlagNeverFired(
                        sensor_id=cfg.id,
                        sessions_observed=sessions_observed,
                    )
                )
            elif fire_rate > cfg.low_signal_threshold.always_fired_rate:
                out.append(
                    SensorSignalFlagAlwaysFiring(
                        sensor_id=cfg.id,
                        fire_rate=fire_rate,
                    )
                )

        def _sort_key(f: SensorSignalFlag) -> tuple[str, str]:
            if isinstance(f, SensorSignalFlagNeverFired):
                return ("a", str(f.sensor_id))
            return ("b", str(f.sensor_id))

        out.sort(key=_sort_key)
        return out


__all__ = [
    "Sensor",
    "SensorAlreadyRegistered",
    "SensorChain",
    "SensorConfig",
    "SensorError",
    "SensorId",
    "SensorInput",
    "SensorKind",
    "SensorOutcome",
    "SensorResult",
    "SensorSignalFlag",
    "SensorSignalFlagAlwaysFiring",
    "SensorSignalFlagNeverFired",
    "SensorSignalThresholds",
    "SensorStats",
    "SensorTrigger",
    "SensorTriggerContinuous",
    "SensorTriggerOnCompaction",
    "SensorTriggerOnToolError",
    "SensorTriggerPostSession",
    "SensorTriggerPostTool",
    "SensorTriggerPostTurn",
    "SensorValidationFailed",
    "StandardSensorChain",
    "new_sensor_id",
]
