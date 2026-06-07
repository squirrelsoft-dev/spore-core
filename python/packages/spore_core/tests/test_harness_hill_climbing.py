"""Tests for the HillClimbing loop strategy (issue #60).

Mirrors ``rust/crates/spore-core/src/harness.rs`` HillClimbing unit tests and
the shared fixture at ``fixtures/metric_evaluator/hill_climbing_sequences.json``.
Each test maps to one pinned spec decision / loop-semantics rule from the issue;
the rule lives in the docstring.

The fixture-replay tests at the bottom read the shared ground-truth JSON and
produce byte-identical TSV for every scenario that embeds ``expected_tsv``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from spore_core import (
    AgentId,
    BudgetLimits,
    FinalResponse,
    HaltReasonBudgetExceeded,
    HaltReasonHillClimbingMisconfigured,
    HaltReasonStagnationLimitReached,
    HarnessConfig,
    HarnessRunOptions,
    HillClimbingConfig,
    MetricErrorCrashed,
    ReactConfig,
    MetricResult,
    MockAgent,
    NoopContextManager,
    RunResultFailure,
    SessionId,
    StandardHarness,
    Task,
    TokenUsage,
)
from spore_core.harness import BaseSandboxProvider, CommandOutput
from spore_core.metric import EvaluateResult
from spore_core.observability import WarnEventHillClimbingIteration
from spore_core.termination import SessionStateSnapshot

FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "fixtures"
    / "metric_evaluator"
    / "hill_climbing_sequences.json"
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class ScriptedEvaluator:
    """Pops scripted metric outcomes in order. Element 0 is the baseline. A
    ``None`` element is a crash (``MetricErrorCrashed``). All ``MetricResult``s
    carry a ZERO duration so the TSV renders ``0.000000`` byte-identically."""

    def __init__(
        self,
        sequence: list[float | None],
        *,
        direction: str = "maximize",
        description: str = "scripted metric",
    ) -> None:
        self._sequence = list(sequence)
        self._idx = 0
        self._direction = direction
        self._description = description
        self.eval_count = 0

    async def evaluate(self, sandbox: Any, session_state: SessionStateSnapshot) -> EvaluateResult:
        self.eval_count += 1
        if self._idx < len(self._sequence):
            element = self._sequence[self._idx]
            self._idx += 1
        else:
            element = None
        if element is None:
            return MetricErrorCrashed(log="scripted crash")
        return MetricResult(value=element, raw_output="", duration=0.0)

    def direction(self) -> str:
        return self._direction

    def description(self) -> str:
        return self._description


class RecordingSandbox(BaseSandboxProvider):
    """Allow-all sandbox over a temp workspace that records every executed
    command, so a test can count ``git reset --hard HEAD`` reverts and read the
    TSV the harness writes."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self.commands: list[tuple[str, list[str]]] = []

    async def execute_command(
        self,
        command: str,
        args: list[str],
        working_dir: Path | None = None,
        timeout: float | None = None,
    ) -> CommandOutput:
        self.commands.append((command, list(args)))
        return CommandOutput(stdout="", stderr="", exit_code=0, timed_out=False)

    def workspace_root(self) -> Path:
        return self._root

    @property
    def revert_count(self) -> int:
        return sum(
            1 for cmd, args in self.commands if cmd == "git" and args == ["reset", "--hard", "HEAD"]
        )


def _agent() -> MockAgent:
    """An agent that always claims done on its turn (one proposed change)."""
    a = MockAgent(AgentId("propose"))
    for _ in range(50):
        a.push(FinalResponse(content="change", usage=TokenUsage(input_tokens=1, output_tokens=1)))
    return a


def _config(
    *,
    sandbox: RecordingSandbox,
    evaluator: Any,
    observability: Any = None,
) -> HarnessConfig:
    from spore_core import AlwaysContinuePolicy, ScriptedToolRegistry

    return HarnessConfig(
        agent=_agent(),
        tool_registry=ScriptedToolRegistry(),
        sandbox=sandbox,
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        metric_evaluator=evaluator,
        observability=observability,
    )


def _task(
    *,
    direction: str = "maximize",
    max_stagnation: int | None = None,
    revert: bool = False,
    min_delta: float | None = None,
    max_turns: int | None = None,
) -> Task:
    return Task.new(
        "optimize the thing",
        SessionId("hc-session"),
        HillClimbingConfig(
            # #124 A.5: the ``inner`` (propose) slot is STRUCTURED — its leaf
            # declares the default output schema handle (empty key).
            inner=ReactConfig(
                budget=ReactConfig.per_loop(2**31 - 1).budget,
                agent="",
                toolset="",
                output="",
            ),
            direction=direction,  # type: ignore[arg-type]
            # ``max_stagnation`` / ``min_improvement_delta`` are required (#119);
            # map the old "unset" sentinels to behavior-preserving values: a
            # never-reached stagnation cap and a zero improvement threshold.
            max_stagnation=max_stagnation if max_stagnation is not None else 2**31 - 1,
            revert_on_no_improvement=revert,
            min_improvement_delta=min_delta if min_delta is not None else 0.0,
            evaluator="",
        ),
        budget=BudgetLimits(max_turns=max_turns),
    )


def _read_tsv(sandbox: RecordingSandbox, task: Task) -> str:
    return (sandbox.workspace_root() / ".spore" / "results" / f"{task.id}.tsv").read_text()


# ---------------------------------------------------------------------------
# Decision 6: a missing evaluator is a typed halt, not a raise.
# ---------------------------------------------------------------------------


async def test_misconfigured_no_evaluator(tmp_path: Path) -> None:
    # #124: with no metric evaluator registered, the unresolved ``evaluator``
    # handle is a STARTUP ConfigurationError (single resolution path) — NOT the
    # legacy HillClimbingMisconfigured. Mirrors Rust's
    # ``hill_climbing_missing_evaluator_is_typed_halt``.
    from spore_core import HaltReasonConfigurationError, HarnessErrorUnresolvedHandle

    sb = RecordingSandbox(tmp_path)
    h = StandardHarness(_config(sandbox=sb, evaluator=None))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonConfigurationError)
    assert r.reason.error == HarnessErrorUnresolvedHandle(handle_kind="metric_evaluator", key="")


# ---------------------------------------------------------------------------
# Decision 7: a baseline (iteration-0) evaluation error is misconfiguration,
# NOT a stagnation increment.
# ---------------------------------------------------------------------------


async def test_baseline_error_is_misconfigured(tmp_path: Path) -> None:
    sb = RecordingSandbox(tmp_path)
    ev = ScriptedEvaluator([None])  # baseline crashes
    task = _task(max_stagnation=3)
    h = StandardHarness(_config(sandbox=sb, evaluator=ev))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonHillClimbingMisconfigured)
    assert "baseline" in r.reason.reason
    # The crashed baseline row is still written, with an empty metric value.
    tsv = _read_tsv(sb, task)
    lines = tsv.splitlines()
    assert lines[0] == (
        "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription"
    )
    # iteration 0, empty commit_hash, EMPTY metric_value, crashed.
    assert lines[1] == "0\t\t\tmaximize\tcrashed\t0.000000\tscripted metric"


# ---------------------------------------------------------------------------
# Decision 5: iteration 0 is a pure baseline (no agent turn), status Kept.
# ---------------------------------------------------------------------------


async def test_baseline_first_kept_no_agent_turn(tmp_path: Path) -> None:
    sb = RecordingSandbox(tmp_path)
    # baseline 1.0 kept, iter1 0.5 discard -> stagnation 1 halts.
    ev = ScriptedEvaluator([1.0, 0.5])
    agent = _agent()
    task = _task(max_stagnation=1)
    from spore_core import AlwaysContinuePolicy, ScriptedToolRegistry

    cfg = HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=sb,
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        metric_evaluator=ev,
    )
    h = StandardHarness(cfg)
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    tsv = _read_tsv(sb, task)
    lines = tsv.splitlines()
    # Row 0 is baseline kept; the agent turn count is 1 (only iter1 ran a turn).
    assert lines[1] == "0\t\t1.000000\tmaximize\tkept\t0.000000\tscripted metric"
    assert r.turns == 1


# ---------------------------------------------------------------------------
# Keep on improve / discard on regress.
# ---------------------------------------------------------------------------


async def test_keep_on_improve(tmp_path: Path) -> None:
    sb = RecordingSandbox(tmp_path)
    ev = ScriptedEvaluator([1.0, 2.0, 1.5])
    task = _task(max_stagnation=1)
    h = StandardHarness(_config(sandbox=sb, evaluator=ev))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonStagnationLimitReached)
    assert r.reason.best_metric == 2.0
    tsv = _read_tsv(sb, task)
    lines = tsv.splitlines()
    assert lines[2] == "1\t\t2.000000\tmaximize\tkept\t0.000000\tscripted metric"
    assert lines[3] == "2\t\t1.500000\tmaximize\tdiscarded\t0.000000\tscripted metric"


# ---------------------------------------------------------------------------
# Strict min_improvement_delta boundary: exactly min_delta is NOT progress.
# ---------------------------------------------------------------------------


async def test_strict_min_delta_boundary(tmp_path: Path) -> None:
    sb = RecordingSandbox(tmp_path)
    # minimize: baseline 2.0, iter1 1.5; improvement of EXACTLY 0.5 == min_delta.
    ev = ScriptedEvaluator([2.0, 1.5], direction="minimize")
    task = _task(direction="minimize", max_stagnation=1, min_delta=0.5)
    h = StandardHarness(_config(sandbox=sb, evaluator=ev))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonStagnationLimitReached)
    assert r.reason.best_metric == 2.0
    lines = _read_tsv(sb, task).splitlines()
    assert lines[2] == "1\t\t1.500000\tminimize\tdiscarded\t0.000000\tscripted metric"


# ---------------------------------------------------------------------------
# Revert on / off.
# ---------------------------------------------------------------------------


async def test_revert_on_regress(tmp_path: Path) -> None:
    sb = RecordingSandbox(tmp_path)
    ev = ScriptedEvaluator([2.0, 3.0], direction="minimize")
    task = _task(direction="minimize", max_stagnation=1, revert=True)
    h = StandardHarness(_config(sandbox=sb, evaluator=ev))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonStagnationLimitReached)
    assert sb.revert_count == 1


async def test_revert_off_regress(tmp_path: Path) -> None:
    sb = RecordingSandbox(tmp_path)
    ev = ScriptedEvaluator([2.0, 3.0], direction="minimize")
    task = _task(direction="minimize", max_stagnation=1, revert=False)
    h = StandardHarness(_config(sandbox=sb, evaluator=ev))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    assert sb.revert_count == 0


# ---------------------------------------------------------------------------
# Stagnation halt + stagnation reset on improve.
# ---------------------------------------------------------------------------


async def test_stagnation_halt_three(tmp_path: Path) -> None:
    sb = RecordingSandbox(tmp_path)
    ev = ScriptedEvaluator([5.0, 4.0, 4.0, 4.0])
    task = _task(max_stagnation=3)
    h = StandardHarness(_config(sandbox=sb, evaluator=ev))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonStagnationLimitReached)
    assert r.reason.iterations == 3
    assert r.reason.best_metric == 5.0


async def test_stagnation_resets_on_improve(tmp_path: Path) -> None:
    sb = RecordingSandbox(tmp_path)
    # baseline 1.0, 0.5 (discard), 0.5 (discard, stag=2), 2.0 (improve, reset),
    # 1.0 (discard, stag=1). max_stagnation 3, max_turns 4 -> ends on turn budget.
    ev = ScriptedEvaluator([1.0, 0.5, 0.5, 2.0, 1.0])
    task = _task(max_stagnation=3, max_turns=4)
    h = StandardHarness(_config(sandbox=sb, evaluator=ev))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonBudgetExceeded)
    assert r.reason.limit_type == "turns"


# ---------------------------------------------------------------------------
# Crash/timeout counts as a non-improvement.
# ---------------------------------------------------------------------------


async def test_crash_counts_as_non_improvement(tmp_path: Path) -> None:
    sb = RecordingSandbox(tmp_path)
    ev = ScriptedEvaluator([1.0, None])
    task = _task(max_stagnation=1)
    h = StandardHarness(_config(sandbox=sb, evaluator=ev))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonStagnationLimitReached)
    assert r.reason.best_metric == 1.0
    lines = _read_tsv(sb, task).splitlines()
    # Crashed row: EMPTY metric_value, status crashed.
    assert lines[2] == "1\t\t\tmaximize\tcrashed\t0.000000\tscripted metric"


# ---------------------------------------------------------------------------
# Budget gate: max_turns 0 means the agent loop never runs an iteration.
# ---------------------------------------------------------------------------


async def test_budget_gate_zero_turns(tmp_path: Path) -> None:
    sb = RecordingSandbox(tmp_path)
    ev = ScriptedEvaluator([1.0, 2.0, 3.0])
    # max_turns 0: baseline runs (no turn), then the gate stops before any agent
    # iteration; clean budget halt. best stays the baseline (no stagnation halt).
    task = _task(max_stagnation=None, max_turns=0)
    h = StandardHarness(_config(sandbox=sb, evaluator=ev))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonBudgetExceeded)
    # Only the baseline was evaluated.
    assert ev.eval_count == 1
    lines = _read_tsv(sb, task).splitlines()
    assert len(lines) == 2  # header + baseline only


# ---------------------------------------------------------------------------
# Observability: a per-iteration span is emitted with metric value/delta.
# ---------------------------------------------------------------------------


async def test_observability_spans(tmp_path: Path) -> None:
    from spore_core.observability import InMemoryObservabilityProvider

    sb = RecordingSandbox(tmp_path)
    obs = InMemoryObservabilityProvider()
    ev = ScriptedEvaluator([1.0, 2.0, 1.5])
    task = _task(max_stagnation=1)
    h = StandardHarness(_config(sandbox=sb, evaluator=ev, observability=obs))
    await h.run(HarnessRunOptions(task))
    spans = await obs.get_trace(SessionId("hc-session"))
    iters = [
        s.event
        for s in spans
        if hasattr(s, "event") and isinstance(s.event, WarnEventHillClimbingIteration)
    ]
    # baseline + 2 iterations = 3 spans.
    assert [e.iteration for e in iters] == [0, 1, 2]
    assert iters[0].delta is None  # baseline has no delta
    assert iters[1].status == "kept"
    assert iters[1].delta == pytest.approx(1.0)
    assert iters[2].status == "discarded"


# ---------------------------------------------------------------------------
# Fixture replay: every scenario reproduces the ground-truth outcome, and the
# scenarios that embed ``expected_tsv`` reproduce it byte-identically.
# ---------------------------------------------------------------------------


def _scenarios() -> list[dict[str, Any]]:
    data = json.loads(FIXTURE.read_text())
    return data["scenarios"]


@pytest.mark.parametrize("scenario", _scenarios(), ids=lambda s: s["name"])
async def test_fixture_replay(scenario: dict[str, Any], tmp_path: Path) -> None:
    payload = scenario["payload"]
    sb = RecordingSandbox(tmp_path)
    ev = ScriptedEvaluator(scenario["metric_sequence"], direction=payload["direction"])
    task = _task(
        direction=payload["direction"],
        max_stagnation=payload["max_stagnation"],
        revert=payload["revert_on_no_improvement"],
        min_delta=payload["min_improvement_delta"],
        max_turns=scenario.get("max_turns"),
    )
    h = StandardHarness(_config(sandbox=sb, evaluator=ev))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)

    expected = scenario["expected"]
    # Halt reason parity.
    if expected["halt_reason"] == "stagnation":
        assert isinstance(r.reason, HaltReasonStagnationLimitReached)
        assert r.reason.best_metric == expected["best_metric"]
    elif expected["halt_reason"] == "budget_turns":
        assert isinstance(r.reason, HaltReasonBudgetExceeded)
        assert r.reason.limit_type == "turns"
    else:
        raise AssertionError(f"unexpected halt_reason: {expected['halt_reason']}")

    # Revert count parity.
    assert sb.revert_count == expected["revert_count"]

    # kept_iterations parity (count of rows recorded as kept).
    tsv = _read_tsv(sb, task)
    kept = sum(1 for line in tsv.splitlines()[1:] if line.split("\t")[4] == "kept")
    assert kept == expected["kept_iterations"]

    # Byte-identical TSV where the scenario embeds it.
    if "expected_tsv" in scenario:
        assert tsv == scenario["expected_tsv"]
