"""Tests for :class:`StandardTerminationPolicy` — issue #13.

Mirrors the Rust unit tests in
``rust/crates/spore-core/src/termination.rs`` ``tests`` module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from spore_core.agent import AgentErrorEmpty
from spore_core.harness import BudgetLimits, BudgetSnapshot, SessionId, TaskId
from spore_core.memory import Timestamp
from spore_core.middleware import HookPoint
from spore_core.sensor import SensorId, SensorOutcome, SensorResult
from spore_core.harness import CommandOutput
from spore_core.model import (
    ContentBlock,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    StopReason,
    TextBlock,
    TextContent,
    TokenUsage,
    ToolCallContent,
    ToolResultContent,
)
from spore_core.termination import (
    AlwaysComplete,
    FeatureListCheck,
    FixedCompletionCheck,
    NullCompletionCheck,
    QuestionAnsweredCheck,
    SessionStateSnapshot,
    SqlResultCheck,
    StandardTerminationPolicy,
    TestSuiteCheck,
    TerminationContinueDecision,
    TerminationDecision,
    TerminationFailureAgentError,
    TerminationFailureCompletionCheckFailed,
    TerminationFailureHumanHalted,
    TerminationFailureMaxRetriesExhausted,
    TerminationFailureMiddlewareHalt,
    TerminationFailurePolicyViolation,
    TerminationFailureReason,
    TerminationFailureUnrecoverableSensorHalt,
    TerminationHaltBudgetExceeded,
    TerminationHaltFailure,
    TerminationHaltSuccess,
    TerminationInput,
    check_budget_default,
)


def _snapshot() -> SessionStateSnapshot:
    return SessionStateSnapshot(
        session_id=SessionId("s1"),
        task_id=TaskId("t1"),
    )


def _input_at(turn: int, done: bool) -> TerminationInput:
    return TerminationInput(
        session_id=SessionId("s1"),
        task_id=TaskId("t1"),
        turn_number=turn,
        agent_claims_done=done,
        agent_response="ok",
        budget_used=BudgetSnapshot(),
        budget_limits=BudgetLimits(),
        sensor_results=[],
        session_state=_snapshot(),
    )


def _sensor_result(sid: str, outcome: SensorOutcome) -> SensorResult:
    return SensorResult(
        sensor_id=SensorId(sid),
        outcome=outcome,
        observation=None,
        detail=outcome.value,
        fired_at=Timestamp("2026-05-17T00:00:00Z"),
    )


# ── Rule: budget is always checked first ─────────────────────────────────


@pytest.mark.asyncio
async def test_budget_hard_stop_when_done() -> None:
    p = StandardTerminationPolicy.with_null_check()
    inp = _input_at(1, True)
    inp.budget_used.turns = 5
    inp.budget_limits.max_turns = 5
    decision = await p.evaluate(inp)
    assert isinstance(decision, TerminationHaltBudgetExceeded)
    assert decision.limit_type == "turns"


@pytest.mark.asyncio
async def test_budget_hard_stop_when_not_done() -> None:
    p = StandardTerminationPolicy.with_null_check()
    inp = _input_at(1, False)
    inp.budget_used.turns = 5
    inp.budget_limits.max_turns = 5
    decision = await p.evaluate(inp)
    assert isinstance(decision, TerminationHaltBudgetExceeded)


def test_budget_check_covers_every_limit_type() -> None:
    cases: list[tuple[BudgetSnapshot, BudgetLimits, str]] = [
        (BudgetSnapshot(turns=3), BudgetLimits(max_turns=3), "turns"),
        (
            BudgetSnapshot(input_tokens=10),
            BudgetLimits(max_input_tokens=10),
            "input_tokens",
        ),
        (
            BudgetSnapshot(output_tokens=10),
            BudgetLimits(max_output_tokens=10),
            "output_tokens",
        ),
        (
            BudgetSnapshot(wall_time=10),
            BudgetLimits(max_wall_time=10),
            "wall_time",
        ),
        (BudgetSnapshot(cost_usd=1.0), BudgetLimits(max_cost_usd=1.0), "cost_usd"),
    ]
    for snap, lim, want in cases:
        got = check_budget_default(snap, lim)
        assert isinstance(got, TerminationHaltBudgetExceeded)
        assert got.limit_type == want
    assert check_budget_default(BudgetSnapshot(), BudgetLimits()) is None


# ── Rule: not-done always continues (after budget) ───────────────────────


@pytest.mark.asyncio
async def test_not_done_continues() -> None:
    p = StandardTerminationPolicy.with_null_check()
    decision = await p.evaluate(_input_at(1, False))
    assert isinstance(decision, TerminationContinueDecision)


# ── Rule: sensor halt becomes UnrecoverableSensorHalt ────────────────────


@pytest.mark.asyncio
async def test_sensor_halt_overrides_completion_success() -> None:
    p = StandardTerminationPolicy.with_null_check()
    inp = _input_at(1, True)
    inp.sensor_results.append(_sensor_result("guardrail", SensorOutcome.HALT))
    decision = await p.evaluate(inp)
    assert isinstance(decision, TerminationHaltFailure)
    assert isinstance(decision.reason, TerminationFailureUnrecoverableSensorHalt)
    assert decision.reason.sensor_id == "guardrail"


@pytest.mark.asyncio
async def test_sensor_warn_does_not_halt() -> None:
    p = StandardTerminationPolicy.with_null_check()
    inp = _input_at(1, True)
    inp.sensor_results.append(_sensor_result("guardrail", SensorOutcome.WARN))
    decision = await p.evaluate(inp)
    assert isinstance(decision, TerminationHaltSuccess)


# ── Rule: completion check returning a reason ⇒ Continue ─────────────────


@pytest.mark.asyncio
async def test_incomplete_check_continues_with_agent_claimed_done() -> None:
    p = StandardTerminationPolicy(FixedCompletionCheck.incomplete("feature B not implemented"))
    decision = await p.evaluate(_input_at(1, True))
    assert isinstance(decision, TerminationContinueDecision)


# ── Rule: completion check returning None ⇒ HaltSuccess(summary) ─────────


@pytest.mark.asyncio
async def test_complete_check_halts_success_with_summary() -> None:
    p = StandardTerminationPolicy.with_null_check()
    inp = _input_at(1, True)
    inp.agent_response = "all green"
    decision = await p.evaluate(inp)
    assert isinstance(decision, TerminationHaltSuccess)
    assert decision.summary == "all green"


@pytest.mark.asyncio
async def test_halt_success_summary_empty_when_no_response() -> None:
    p = StandardTerminationPolicy.with_null_check()
    inp = _input_at(1, True)
    inp.agent_response = None
    decision = await p.evaluate(inp)
    assert isinstance(decision, TerminationHaltSuccess)
    assert decision.summary == ""


# ── Rule: HaltFailure carries typed reason ────────────────────────────────


def test_halt_failure_reason_is_typed() -> None:
    """Round-trip every variant through JSON to prove the wire format
    matches the cross-language fixture schema."""
    reason_adapter: TypeAdapter[TerminationFailureReason] = TypeAdapter(TerminationFailureReason)
    cases: list[TerminationFailureReason] = [
        TerminationFailureCompletionCheckFailed(detail="nope"),
        TerminationFailureMaxRetriesExhausted(tool="bash", attempts=3),
        TerminationFailureUnrecoverableSensorHalt(sensor_id=SensorId("g"), detail="tripped"),
        TerminationFailureMiddlewareHalt(hook=HookPoint.BEFORE_TURN, reason="veto"),
        TerminationFailureAgentError(error=AgentErrorEmpty()),
        TerminationFailurePolicyViolation(detail="policy"),
        TerminationFailureHumanHalted(),
    ]
    for c in cases:
        as_json = c.model_dump_json()
        back = reason_adapter.validate_json(as_json)
        assert back == c


# ── Fixture replay ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fixture_replay_basic() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[4] / "fixtures" / "termination_policy" / "basic.json"
    )
    suite = json.loads(fixture_path.read_text())
    decision_adapter: TypeAdapter[TerminationDecision] = TypeAdapter(TerminationDecision)
    for case in suite["cases"]:
        cc_spec = case["completion_check"]
        if cc_spec["kind"] == "complete":
            check = FixedCompletionCheck.complete()
        elif cc_spec["kind"] == "incomplete":
            check = FixedCompletionCheck.incomplete(cc_spec["reason"])
        else:
            raise AssertionError(f"unknown completion check: {cc_spec!r}")

        policy = StandardTerminationPolicy(check)
        sensor_results = [SensorResult.model_validate(r) for r in case.get("sensor_results", [])]
        inp = TerminationInput(
            session_id=SessionId("fixture"),
            task_id=TaskId("fixture-task"),
            turn_number=1,
            agent_claims_done=case["agent_claims_done"],
            agent_response=case.get("agent_response"),
            budget_used=BudgetSnapshot.model_validate(case["budget_used"]),
            budget_limits=BudgetLimits.model_validate(case["budget_limits"]),
            sensor_results=sensor_results,
            session_state=_snapshot(),
        )
        got = await policy.evaluate(inp)
        got_json = json.loads(decision_adapter.dump_json(got).decode("utf-8"))
        assert got_json == case["expected"], (
            f"fixture case `{case['name']}` produced unexpected decision: {got_json!r}"
        )


# =========================================================================
# FeatureListCheck (issue #43)
# =========================================================================


def _snapshot_in(workspace: Path) -> SessionStateSnapshot:
    return SessionStateSnapshot(
        session_id=SessionId("s1"),
        task_id=TaskId("t1"),
        workspace_root=workspace,
    )


@pytest.mark.asyncio
async def test_feature_list_all_pass_returns_none(tmp_path: Path) -> None:
    (tmp_path / "feature_list.json").write_text(
        '[{"name":"a","passes":true},{"name":"b","passes":true}]'
    )
    assert await FeatureListCheck().check(_snapshot_in(tmp_path)) is None


@pytest.mark.asyncio
async def test_feature_list_some_fail_returns_reason(tmp_path: Path) -> None:
    (tmp_path / "feature_list.json").write_text(
        '[{"name":"a","passes":true},{"name":"b","passes":false},{"name":"c","passes":false}]'
    )
    r = await FeatureListCheck().check(_snapshot_in(tmp_path))
    assert r is not None
    assert "b" in r and "c" in r


@pytest.mark.asyncio
async def test_feature_list_missing_file_returns_reason(tmp_path: Path) -> None:
    r = await FeatureListCheck().check(_snapshot_in(tmp_path))
    assert r is not None
    assert "missing" in r


@pytest.mark.asyncio
async def test_feature_list_custom_path(tmp_path: Path) -> None:
    (tmp_path / "custom.json").write_text('[{"name":"x","passes":true}]')
    assert await FeatureListCheck("custom.json").check(_snapshot_in(tmp_path)) is None


# =========================================================================
# TestSuiteCheck (issue #43)
# =========================================================================


class _StubSandbox:
    """Minimal SandboxProvider stub for TestSuiteCheck."""

    def __init__(self, exit_code: int, stderr: str = "", stdout: str = "") -> None:
        self._out = CommandOutput(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=False,
            truncated=False,
        )
        self._root = Path("/")

    async def validate(self, call):  # type: ignore[no-untyped-def]
        return None

    async def execute_command(
        self,
        command: str,
        args: list[str],
        working_dir: Path | None = None,
        timeout: float | None = None,
    ) -> CommandOutput:
        return self._out

    async def handle_large_output(self, content, call_id, head_tokens, tail_tokens):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def resolve_path(self, path: str, operation: str = "read") -> Path:
        return Path(path)

    def isolation_mode(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def workspace_root(self) -> Path:
        return self._root


@pytest.mark.asyncio
async def test_test_suite_pass_returns_none() -> None:
    check = TestSuiteCheck("cargo test", Path("."), 30.0, _StubSandbox(0))
    assert await check.check(_snapshot()) is None


@pytest.mark.asyncio
async def test_test_suite_fail_includes_stderr_tail() -> None:
    sandbox = _StubSandbox(1, stderr="test foo ... FAILED\nassertion failed")
    check = TestSuiteCheck("cargo test", Path("."), 30.0, sandbox)
    r = await check.check(_snapshot())
    assert r is not None and "FAILED" in r


@pytest.mark.asyncio
async def test_test_suite_empty_command_fails_cleanly() -> None:
    check = TestSuiteCheck("", Path("."), 30.0, _StubSandbox(0))
    assert await check.check(_snapshot()) is not None


# =========================================================================
# QuestionAnsweredCheck (issue #43)
# =========================================================================


class _StubJudge:
    def __init__(self, verdict: str) -> None:
        self._verdict = verdict

    async def call(self, request: ModelRequest) -> ModelResponse:
        content: list[ContentBlock] = [TextBlock(text=self._verdict)]
        return ModelResponse(
            content=content,
            usage=TokenUsage(),
            stop_reason=StopReason.END_TURN,
        )

    def call_streaming(self, request: ModelRequest):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def count_tokens(self, request: ModelRequest) -> int:
        return 0

    def provider(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def _snap_with_assistant(text: str) -> SessionStateSnapshot:
    snap = _snapshot()
    snap.state.messages.append(Message(role=Role.ASSISTANT, content=TextContent(text=text)))
    return snap


@pytest.mark.asyncio
async def test_question_answered_yes_returns_none() -> None:
    judge = _StubJudge("ANSWERED: YES\nLooks good.")
    c = QuestionAnsweredCheck(judge, "What is 2+2?")
    snap = _snap_with_assistant("It is 4.")
    assert await c.check(snap) is None


@pytest.mark.asyncio
async def test_question_answered_no_returns_reason() -> None:
    judge = _StubJudge("ANSWERED: NO\nMissed the point.")
    c = QuestionAnsweredCheck(judge, "What is 2+2?")
    snap = _snap_with_assistant("I don't know.")
    r = await c.check(snap)
    assert r is not None and "not answered" in r


@pytest.mark.asyncio
async def test_question_answered_uses_rubric() -> None:
    judge = _StubJudge("ANSWERED: YES")
    c = QuestionAnsweredCheck(judge, "q").with_rubric("Be strict about citations.")
    assert "q" in c.description()
    assert await c.check(_snap_with_assistant("a")) is None


# =========================================================================
# SqlResultCheck (issue #43)
# =========================================================================


def _snap_with_sql_result(tool_name: str, body: str) -> SessionStateSnapshot:
    snap = _snapshot()
    snap.state.messages.append(
        Message(
            role=Role.ASSISTANT,
            content=ToolCallContent(id="call-1", name=tool_name, input={"q": "select 1"}),
        )
    )
    snap.state.messages.append(
        Message(
            role=Role.TOOL,
            content=ToolResultContent(tool_use_id="call-1", content=body, is_error=False),
        )
    )
    return snap


@pytest.mark.asyncio
async def test_sql_result_check_default_passes_when_rows_present() -> None:
    snap = _snap_with_sql_result(
        "execute_sql",
        '{"columns":["id","name"],"rows":[[1,"a"],[2,"b"]]}',
    )
    assert await SqlResultCheck().check(snap) is None


@pytest.mark.asyncio
async def test_sql_result_check_empty_rows_fails() -> None:
    snap = _snap_with_sql_result("execute_sql", '{"columns":["id"],"rows":[]}')
    r = await SqlResultCheck().check(snap)
    assert r is not None and "0 rows" in r


@pytest.mark.asyncio
async def test_sql_result_check_column_mismatch_fails() -> None:
    snap = _snap_with_sql_result("execute_sql", '{"columns":["id"],"rows":[[1]]}')
    r = await SqlResultCheck().with_expected_columns(["id", "name"]).check(snap)
    assert r is not None and "columns mismatch" in r


@pytest.mark.asyncio
async def test_sql_result_check_min_rows_enforced() -> None:
    snap = _snap_with_sql_result("execute_sql", '{"columns":["id"],"rows":[[1]]}')
    r = await SqlResultCheck().with_min_rows(5).check(snap)
    assert r is not None and "at least 5" in r


@pytest.mark.asyncio
async def test_sql_result_check_custom_tool_name() -> None:
    snap = _snap_with_sql_result("run_query", '{"columns":[],"rows":[[1]]}')
    c = SqlResultCheck().with_tool_name("run_query")
    assert await c.check(snap) is None


@pytest.mark.asyncio
async def test_sql_result_check_no_matching_tool_fails() -> None:
    snap = _snap_with_sql_result("other_tool", '{"columns":[],"rows":[[1]]}')
    r = await SqlResultCheck().check(snap)
    assert r is not None and "no `execute_sql`" in r


# =========================================================================
# SqlResultCheck — cross-language fixture replay
# =========================================================================


@pytest.mark.asyncio
async def test_fixture_replay_sql_result_check() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[4] / "fixtures" / "completion_check" / "sql_result.json"
    )
    suite = json.loads(fixture_path.read_text())
    for case in suite["cases"]:
        messages = [Message.model_validate(m) for m in case["messages"]]
        snap = SessionStateSnapshot(
            session_id=SessionId("fix"),
            task_id=TaskId("fix"),
        )
        snap.state.messages = messages
        check = SqlResultCheck().with_tool_name(case["sql_tool_name"])
        if case.get("expected_columns") is not None:
            check = check.with_expected_columns(case["expected_columns"])
        if case.get("min_rows") is not None:
            check = check.with_min_rows(case["min_rows"])
        got = await check.check(snap)
        expected = case["expected"]
        if expected["kind"] == "complete":
            assert got is None, f"case `{case['name']}`: expected complete, got `{got}`"
        elif expected["kind"] == "incomplete":
            assert got is not None and expected["contains"] in got, (
                f"case `{case['name']}`: expected reason to contain "
                f"`{expected['contains']}`, got `{got}`"
            )
        else:
            raise AssertionError(f"unknown expected kind: {expected['kind']}")


# =========================================================================
# AlwaysComplete alias
# =========================================================================


@pytest.mark.asyncio
async def test_always_complete_is_null_check() -> None:
    a = AlwaysComplete()
    assert isinstance(a, NullCompletionCheck)
    assert await a.check(_snapshot()) is None
