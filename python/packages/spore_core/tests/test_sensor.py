"""Tests for :class:`StandardSensorChain` — issue #10.

Mirrors the Rust unit tests in
``rust/crates/spore-core/src/sensor.rs`` ``tests`` module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spore_core.harness import SessionId
from spore_core.memory import Timestamp
from spore_core.sensor import (
    Sensor,
    SensorAlreadyRegistered,
    SensorConfig,
    SensorId,
    SensorInput,
    SensorKind,
    SensorOutcome,
    SensorResult,
    SensorSignalFlagAlwaysFiring,
    SensorSignalFlagNeverFired,
    SensorSignalThresholds,
    SensorTriggerPostSession,
    SensorTriggerPostTool,
    SensorTriggerPostTurn,
    SensorValidationFailed,
    StandardSensorChain,
)
from spore_core.tool_registry import TaskPhase


# ── Programmable stub sensor ───────────────────────────────────────────────


class StubSensor:
    """Programmable sensor used across tests."""

    def __init__(self, cfg: SensorConfig, outcome: SensorOutcome) -> None:
        self._cfg = cfg
        self._outcome = outcome

    async def evaluate(self, input: SensorInput) -> SensorResult:
        return SensorResult(
            sensor_id=self._cfg.id,
            outcome=self._outcome,
            observation="warn-obs" if self._outcome == SensorOutcome.WARN else None,
            detail=self._outcome.value,
            fired_at=Timestamp("2026-05-16T00:00:00Z"),
        )

    def config(self) -> SensorConfig:
        return self._cfg


def _computational(
    id_: str,
    triggers: list,
    outcome: SensorOutcome,
    *,
    thresholds: SensorSignalThresholds | None = None,
    run_every_n_turns: int | None = None,
) -> StubSensor:
    cfg = SensorConfig(
        id=SensorId(id_),
        name=id_,
        kind=SensorKind.COMPUTATIONAL,
        triggers=triggers,
        run_every_n_turns=run_every_n_turns,
        run_on_phases=None,
        low_signal_threshold=thresholds or SensorSignalThresholds(),
    )
    return StubSensor(cfg, outcome)


def _inferential(
    id_: str,
    triggers: list,
    outcome: SensorOutcome,
    *,
    every_n: int | None = None,
    phases: list[TaskPhase] | None = None,
) -> StubSensor:
    cfg = SensorConfig(
        id=SensorId(id_),
        name=id_,
        kind=SensorKind.INFERENTIAL,
        triggers=triggers,
        run_every_n_turns=every_n,
        run_on_phases=phases,
    )
    return StubSensor(cfg, outcome)


def _input(sid: str, *, turn: int | None = None, phase: TaskPhase | None = None) -> SensorInput:
    return SensorInput(session_id=SessionId(sid), turn_number=turn, phase=phase)


# ── Rule: register validates triggers ──────────────────────────────────────


async def test_register_rejects_empty_triggers() -> None:
    chain = StandardSensorChain()
    s = _computational("s1", [], SensorOutcome.PASS)
    with pytest.raises(SensorValidationFailed):
        await chain.register(s)


async def test_register_rejects_duplicate_ids() -> None:
    chain = StandardSensorChain()
    await chain.register(_computational("s1", [SensorTriggerPostTurn()], SensorOutcome.PASS))
    with pytest.raises(SensorAlreadyRegistered):
        await chain.register(_computational("s1", [SensorTriggerPostTurn()], SensorOutcome.PASS))


# ── Rule: fire runs every matching sensor, returns all results ─────────────


async def test_fire_runs_all_matching_sensors_no_short_circuit() -> None:
    chain = StandardSensorChain()
    await chain.register(_computational("pass", [SensorTriggerPostTurn()], SensorOutcome.PASS))
    await chain.register(_computational("warn", [SensorTriggerPostTurn()], SensorOutcome.WARN))
    await chain.register(_computational("halt", [SensorTriggerPostTurn()], SensorOutcome.HALT))
    results = await chain.fire(SensorTriggerPostTurn(), _input("s1"))
    assert len(results) == 3
    outcomes = {r.outcome for r in results}
    assert outcomes == {SensorOutcome.PASS, SensorOutcome.WARN, SensorOutcome.HALT}


# ── Rule: triggers filter ──────────────────────────────────────────────────


async def test_fire_ignores_sensors_without_matching_trigger() -> None:
    chain = StandardSensorChain()
    await chain.register(_computational("post-turn", [SensorTriggerPostTurn()], SensorOutcome.PASS))
    await chain.register(
        _computational("post-session", [SensorTriggerPostSession()], SensorOutcome.PASS)
    )
    results = await chain.fire(SensorTriggerPostTurn(), _input("s1"))
    assert len(results) == 1
    assert results[0].sensor_id == SensorId("post-turn")


# ── Rule: PostTool wildcard matches any tool, named matches exact ──────────


async def test_post_tool_wildcard_and_named_matching() -> None:
    chain = StandardSensorChain()
    await chain.register(
        _computational("any", [SensorTriggerPostTool(tool_name="")], SensorOutcome.PASS)
    )
    await chain.register(
        _computational("bash-only", [SensorTriggerPostTool(tool_name="bash")], SensorOutcome.PASS)
    )
    r1 = await chain.fire(SensorTriggerPostTool(tool_name="bash"), _input("s1"))
    assert len(r1) == 2
    r2 = await chain.fire(SensorTriggerPostTool(tool_name="edit"), _input("s2"))
    assert len(r2) == 1
    assert r2[0].sensor_id == SensorId("any")


# ── Rule: Computational sensors ignore run_every_n_turns ───────────────────


async def test_computational_ignores_turn_gating() -> None:
    chain = StandardSensorChain()
    s = _computational("c", [SensorTriggerPostTurn()], SensorOutcome.PASS, run_every_n_turns=99)
    await chain.register(s)
    results = await chain.fire(SensorTriggerPostTurn(), _input("s1", turn=1))
    assert len(results) == 1


# ── Rule: Inferential gated by run_every_n_turns ───────────────────────────


async def test_inferential_run_every_n_turns_gating() -> None:
    chain = StandardSensorChain()
    await chain.register(
        _inferential("judge", [SensorTriggerPostTurn()], SensorOutcome.WARN, every_n=3)
    )
    assert await chain.fire(SensorTriggerPostTurn(), _input("s1", turn=1)) == []
    assert await chain.fire(SensorTriggerPostTurn(), _input("s1", turn=2)) == []
    assert len(await chain.fire(SensorTriggerPostTurn(), _input("s1", turn=3))) == 1
    assert len(await chain.fire(SensorTriggerPostTurn(), _input("s1", turn=6))) == 1


# ── Rule: Inferential gated by run_on_phases ───────────────────────────────


async def test_inferential_run_on_phases_gating() -> None:
    chain = StandardSensorChain()
    await chain.register(
        _inferential(
            "judge",
            [SensorTriggerPostTurn()],
            SensorOutcome.PASS,
            phases=[TaskPhase.EXECUTION],
        )
    )
    assert await chain.fire(SensorTriggerPostTurn(), _input("s1", phase=TaskPhase.PLANNING)) == []
    assert (
        len(await chain.fire(SensorTriggerPostTurn(), _input("s1", phase=TaskPhase.EXECUTION))) == 1
    )


# ── Rule: stats aggregates per-sensor outcomes ─────────────────────────────


async def test_stats_aggregates_outcomes_and_fire_rate() -> None:
    chain = StandardSensorChain()
    await chain.register(_computational("warner", [SensorTriggerPostTurn()], SensorOutcome.WARN))
    for i in range(4):
        await chain.fire(SensorTriggerPostTurn(), _input(f"s{i}"))
    stats = await chain.stats()
    assert len(stats) == 1
    s = stats[0]
    assert s.total_fires == 4
    assert s.warn_count == 4
    assert s.halt_count == 0
    assert s.pass_count == 0
    assert abs(s.fire_rate - 1.0) < 1e-6


# ── Rule: signal_quality_report flags AlwaysFiring ─────────────────────────


async def test_signal_quality_flags_always_firing() -> None:
    chain = StandardSensorChain()
    s = _computational(
        "noisy",
        [SensorTriggerPostTurn()],
        SensorOutcome.WARN,
        thresholds=SensorSignalThresholds(never_fired_after_n_sessions=100, always_fired_rate=0.5),
    )
    await chain.register(s)
    for i in range(5):
        await chain.fire(SensorTriggerPostTurn(), _input(f"s{i}"))
    flags = await chain.signal_quality_report(5)
    assert any(
        isinstance(f, SensorSignalFlagAlwaysFiring) and f.sensor_id == SensorId("noisy")
        for f in flags
    )


# ── Rule: signal_quality_report flags NeverFired ───────────────────────────


async def test_signal_quality_flags_never_fired() -> None:
    chain = StandardSensorChain()
    s = _computational(
        "quiet",
        [SensorTriggerPostSession()],
        SensorOutcome.PASS,
        thresholds=SensorSignalThresholds(never_fired_after_n_sessions=3, always_fired_rate=0.9),
    )
    await chain.register(s)
    for i in range(5):
        await chain.fire(SensorTriggerPostTurn(), _input(f"s{i}"))
    flags = await chain.signal_quality_report(3)
    assert any(
        isinstance(f, SensorSignalFlagNeverFired)
        and f.sensor_id == SensorId("quiet")
        and f.sessions_observed >= 3
        for f in flags
    )


# ── Rule: signal_quality_report respects min_sessions floor ────────────────


async def test_signal_quality_respects_min_sessions() -> None:
    chain = StandardSensorChain()
    await chain.register(_computational("quiet", [SensorTriggerPostSession()], SensorOutcome.PASS))
    await chain.fire(SensorTriggerPostTurn(), _input("s1"))
    flags = await chain.signal_quality_report(10)
    assert flags == []


# ── Fixture replay ─────────────────────────────────────────────────────────


async def test_fixture_replay_signal_quality() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[4]
        / "fixtures"
        / "sensor_chain"
        / "signal_quality_basic.json"
    )
    case = json.loads(fixture_path.read_text())

    chain = StandardSensorChain()
    for spec in case["sensors"]:
        trigger_objs = []
        for t in spec["triggers"]:
            k = t["kind"]
            if k == "post_turn":
                trigger_objs.append(SensorTriggerPostTurn())
            elif k == "post_session":
                trigger_objs.append(SensorTriggerPostSession())
            elif k == "post_tool":
                trigger_objs.append(SensorTriggerPostTool(tool_name=t.get("tool_name", "")))
            else:
                raise AssertionError(f"unknown trigger kind: {k}")
        thresholds = SensorSignalThresholds(**spec["thresholds"])
        cfg = SensorConfig(
            id=SensorId(spec["id"]),
            name=spec["id"],
            kind=SensorKind(spec["kind"]),
            triggers=trigger_objs,
            low_signal_threshold=thresholds,
        )
        await chain.register(StubSensor(cfg, SensorOutcome(spec["outcome"])))

    for ev in case["events"]:
        k = ev["trigger"]["kind"]
        if k == "post_turn":
            trig = SensorTriggerPostTurn()
        elif k == "post_session":
            trig = SensorTriggerPostSession()
        elif k == "post_tool":
            trig = SensorTriggerPostTool(tool_name=ev["trigger"].get("tool_name", ""))
        else:
            raise AssertionError(f"unknown trigger kind: {k}")
        await chain.fire(trig, _input(ev["session_id"]))

    flags = await chain.signal_quality_report(case["min_sessions"])
    got_never = {str(f.sensor_id) for f in flags if isinstance(f, SensorSignalFlagNeverFired)}
    got_always = {str(f.sensor_id) for f in flags if isinstance(f, SensorSignalFlagAlwaysFiring)}
    assert got_never == set(case["expected"]["never_fired"])
    assert got_always == set(case["expected"]["always_firing"])


# ── Sensor protocol structural conformance ─────────────────────────────────


def test_stub_sensor_satisfies_protocol() -> None:
    s = _computational("x", [SensorTriggerPostTurn()], SensorOutcome.PASS)
    assert isinstance(s, Sensor)
