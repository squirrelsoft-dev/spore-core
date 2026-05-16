"""Tests for Harness runtime loop (issue #3).

Mirrors ``rust/crates/spore-core/src/harness.rs`` unit tests. Each test
exercises one rule from the spec; rule lives in the test docstring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from spore_core import (
    AgentErrorEmpty,
    AgentId,
    AggregateUsage,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetSnapshot,
    ChildPausedState,
    FinalResponse,
    HaltReasonAgentError,
    HaltReasonBudgetExceeded,
    HaltReasonHumanHalted,
    HaltReasonMiddlewareHalt,
    HaltReasonSandboxViolation,
    HaltReasonStrategyNotYetImplemented,
    HaltReasonTerminationPolicyHalt,
    HaltReasonUnrecoverableToolError,
    HarnessConfig,
    HarnessRunOptions,
    HumanRequestClarification,
    HumanRequestToolApproval,
    HumanResponseAllow,
    HumanResponseHalt,
    LoopStrategyHillClimbing,
    LoopStrategyPlanExecute,
    LoopStrategyRalph,
    LoopStrategyReAct,
    LoopStrategySelfVerifying,
    MiddlewareHalt,
    MiddlewareSurfaceToHuman,
    MockAgent,
    NoopContextManager,
    PausedState,
    ProviderInfo,
    ReplayModelInterface,
    RunResult,
    RunResultFailure,
    RunResultSuccess,
    RunResultWaitingForHuman,
    SandboxPathDenied,
    SandboxPathEscape,
    ScriptedMiddleware,
    ScriptedSandbox,
    ScriptedTerminationPolicy,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    StandardHarness,
    Task,
    TaskId,
    TerminationHalt,
    TokenUsage,
    ToolCall,
    ToolCallRequested,
    ToolOutputError,
    ToolOutputSuccess,
    ToolOutputWaitingForHuman,
    TurnError,
)
from spore_core.agent import ModelAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent() -> MockAgent:
    return MockAgent(AgentId("test"))


def _config(agent: MockAgent, **overrides: Any) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=overrides.get("tool_registry", ScriptedToolRegistry()),
        sandbox=overrides.get("sandbox", AllowAllSandbox()),
        context_manager=overrides.get("context_manager", NoopContextManager()),
        termination_policy=overrides.get("termination_policy", AlwaysContinuePolicy()),
        middleware=overrides.get("middleware"),
        observability=overrides.get("observability"),
    )


def _react_task(max_iter: int = 5) -> Task:
    return Task.new(
        "do something",
        SessionId("s1"),
        LoopStrategyReAct(max_iterations=max_iter),
    )


def _usage(in_t: int = 1, out_t: int = 1) -> TokenUsage:
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


def _tc(call_id: str = "c", name: str = "x") -> ToolCall:
    return ToolCall(id=call_id, name=name, input={})


# ---------------------------------------------------------------------------
# Rule: Harness owns the loop; final response on first turn returns Success.
# ---------------------------------------------------------------------------


async def test_final_response_returns_success() -> None:
    a = _agent()
    a.push(FinalResponse(content="done", usage=_usage()))
    h = StandardHarness(_config(a))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "done"
    assert r.turns == 1


# ---------------------------------------------------------------------------
# Rule: tool calls are dispatched, then loop continues to a final response.
# ---------------------------------------------------------------------------


async def test_tool_call_then_final_response_loops() -> None:
    a = _agent()
    a.push(ToolCallRequested(calls=[_tc("c1", "x")], usage=_usage()))
    a.push(FinalResponse(content="after-tool", usage=_usage()))
    reg = ScriptedToolRegistry().push(ToolOutputSuccess(content="tool-ok"))
    h = StandardHarness(_config(a, tool_registry=reg))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "after-tool"
    assert r.turns == 2
    assert reg.call_count == 1


# ---------------------------------------------------------------------------
# Rule: parallel tool calls all dispatched in one turn.
# ---------------------------------------------------------------------------


async def test_parallel_tool_calls_all_dispatched() -> None:
    a = _agent()
    a.push(
        ToolCallRequested(
            calls=[_tc("a", "x"), _tc("b", "y")],
            usage=_usage(),
        )
    )
    a.push(FinalResponse(content="ok", usage=_usage()))
    reg = ScriptedToolRegistry()
    reg.push(ToolOutputSuccess(content="1"))
    reg.push(ToolOutputSuccess(content="2"))
    h = StandardHarness(_config(a, tool_registry=reg))
    await h.run(HarnessRunOptions(_react_task()))
    assert reg.call_count == 2


# ---------------------------------------------------------------------------
# Rule: budget overrun terminates with explicit reason.
# ---------------------------------------------------------------------------


async def test_budget_max_turns_exceeded() -> None:
    a = _agent()
    for _ in range(10):
        a.push(ToolCallRequested(calls=[_tc()], usage=_usage()))
    h = StandardHarness(_config(a))
    t = _react_task(100)
    t.budget.max_turns = 2
    r = await h.run(HarnessRunOptions(t))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonBudgetExceeded)
    assert r.reason.limit_type == "turns"
    assert r.turns == 2


async def test_budget_max_input_tokens_exceeded() -> None:
    """Rule: input-token budget overrun is reported explicitly."""
    a = _agent()
    a.push(ToolCallRequested(calls=[_tc()], usage=_usage(in_t=100, out_t=1)))
    a.push(FinalResponse(content="x", usage=_usage()))
    reg = ScriptedToolRegistry().push(ToolOutputSuccess(content="ok"))
    h = StandardHarness(_config(a, tool_registry=reg))
    t = _react_task()
    t.budget.max_input_tokens = 50
    r = await h.run(HarnessRunOptions(t))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonBudgetExceeded)
    assert r.reason.limit_type == "input_tokens"


# ---------------------------------------------------------------------------
# Rule: A turn with neither tool call nor response is an error.
# ---------------------------------------------------------------------------


async def test_agent_error_terminates_with_agent_error_halt_reason() -> None:
    a = _agent()
    a.push(TurnError(error=AgentErrorEmpty(), usage=_usage()))
    h = StandardHarness(_config(a))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonAgentError)


# ---------------------------------------------------------------------------
# Rule: Layer-1 SandboxViolation::PathEscape halts unconditionally.
# ---------------------------------------------------------------------------


async def test_layer1_path_escape_halts() -> None:
    a = _agent()
    a.push(ToolCallRequested(calls=[_tc(name="read")], usage=_usage()))
    sb = ScriptedSandbox().push(SandboxPathEscape(path="/etc/passwd"))
    h = StandardHarness(_config(a, sandbox=sb))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonSandboxViolation)
    assert isinstance(r.reason.violation, SandboxPathEscape)


# ---------------------------------------------------------------------------
# Rule: Layer-2 recoverable sandbox violation → appended as tool error,
# loop continues.
# ---------------------------------------------------------------------------


async def test_layer2_path_denied_continues_as_tool_error() -> None:
    a = _agent()
    a.push(ToolCallRequested(calls=[_tc(name="read")], usage=_usage()))
    a.push(FinalResponse(content="ack", usage=_usage()))
    sb = ScriptedSandbox().push(SandboxPathDenied(path="/p"))
    h = StandardHarness(_config(a, sandbox=sb))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.turns == 2


# ---------------------------------------------------------------------------
# Rule: TerminationPolicy::Halt overrides final response.
# ---------------------------------------------------------------------------


async def test_termination_policy_halt_overrides_success() -> None:
    a = _agent()
    a.push(FinalResponse(content="done", usage=_usage()))
    tp = ScriptedTerminationPolicy().push(TerminationHalt(reason="not yet"))
    h = StandardHarness(_config(a, termination_policy=tp))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonTerminationPolicyHalt)
    assert r.reason.reason == "not yet"


# ---------------------------------------------------------------------------
# Rule: Middleware::Halt at BeforeTurn halts before model call.
# ---------------------------------------------------------------------------


async def test_middleware_halt_before_turn() -> None:
    a = _agent()
    a.push(FinalResponse(content="unused", usage=_usage()))
    mw = ScriptedMiddleware().push("before_turn", MiddlewareHalt(reason="blocked"))
    h = StandardHarness(_config(a, middleware=mw))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonMiddlewareHalt)
    assert r.reason.hook == "before_turn"
    assert r.reason.reason == "blocked"
    assert r.turns == 0


# ---------------------------------------------------------------------------
# Rule: Middleware::SurfaceToHuman at BeforeTool returns WaitingForHuman
# with pending calls preserved.
# ---------------------------------------------------------------------------


async def test_middleware_surface_to_human_before_tool() -> None:
    a = _agent()
    calls = [_tc()]
    a.push(ToolCallRequested(calls=calls, usage=_usage()))
    req = HumanRequestToolApproval(calls=calls, risk_level="medium")
    mw = ScriptedMiddleware().push("before_tool", MiddlewareSurfaceToHuman(request=req))
    h = StandardHarness(_config(a, middleware=mw))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultWaitingForHuman)
    assert len(r.state.pending_tool_calls) == 1
    assert r.state.child_state is None


# ---------------------------------------------------------------------------
# Rule: always_halt tool annotation halts the loop.
# ---------------------------------------------------------------------------


async def test_always_halt_tool_halts() -> None:
    a = _agent()
    a.push(ToolCallRequested(calls=[_tc(name="danger")], usage=_usage()))
    reg = ScriptedToolRegistry()
    reg.mark_always_halt("danger")
    h = StandardHarness(_config(a, tool_registry=reg))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonUnrecoverableToolError)
    assert r.reason.tool == "danger"


# ---------------------------------------------------------------------------
# Rule: Unrecoverable tool error halts loop immediately.
# ---------------------------------------------------------------------------


async def test_unrecoverable_tool_error_halts() -> None:
    a = _agent()
    a.push(ToolCallRequested(calls=[_tc()], usage=_usage()))
    reg = ScriptedToolRegistry().push(ToolOutputError(message="boom", recoverable=False))
    h = StandardHarness(_config(a, tool_registry=reg))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonUnrecoverableToolError)
    assert r.reason.error == "boom"


# ---------------------------------------------------------------------------
# Rule: WaitingForHuman from a tool dispatch propagates to RunResult and
# records child_state.
# ---------------------------------------------------------------------------


async def test_tool_waiting_for_human_propagates() -> None:
    a = _agent()
    a.push(ToolCallRequested(calls=[_tc(name="subagent")], usage=_usage()))
    child_task = _react_task(1)
    cps = ChildPausedState(
        session_id=SessionId("child"),
        task_id=TaskId("ct"),
        turn_number=1,
        session_state=SessionState(),
        pending_tool_calls=[],
        approved_results=[],
        human_request=HumanRequestClarification(question="?"),
        task=child_task,
        budget_used=BudgetSnapshot(),
        parent_tool_call_id="c",
    )
    reg = ScriptedToolRegistry().push(
        ToolOutputWaitingForHuman(
            child_state=cps,
            request=HumanRequestClarification(question="?"),
        )
    )
    h = StandardHarness(_config(a, tool_registry=reg))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultWaitingForHuman)
    assert r.state.child_state is not None


# ---------------------------------------------------------------------------
# Rule: resume() with Halt returns Failure(HumanHalted).
# ---------------------------------------------------------------------------


async def test_resume_with_halt_returns_human_halted() -> None:
    a = _agent()
    h = StandardHarness(_config(a))
    state = PausedState(
        session_id=SessionId("s"),
        task_id=TaskId("t"),
        turn_number=1,
        session_state=SessionState(),
        pending_tool_calls=[],
        approved_results=[],
        human_request=HumanRequestClarification(question="?"),
        task=_react_task(),
        budget_used=BudgetSnapshot(),
        child_state=None,
    )
    r = await h.resume(state, HumanResponseHalt())
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonHumanHalted)


# ---------------------------------------------------------------------------
# Rule: resume() with Allow dispatches pending tool calls then continues loop.
# ---------------------------------------------------------------------------


async def test_resume_with_allow_executes_pending_and_continues() -> None:
    a = _agent()
    a.push(FinalResponse(content="done", usage=_usage()))
    reg = ScriptedToolRegistry().push(ToolOutputSuccess(content="tool-ok"))
    h = StandardHarness(_config(a, tool_registry=reg))
    state = PausedState(
        session_id=SessionId("s"),
        task_id=TaskId("t"),
        turn_number=1,
        session_state=SessionState(),
        pending_tool_calls=[_tc()],
        approved_results=[],
        human_request=HumanRequestToolApproval(calls=[], risk_level="low"),
        task=_react_task(),
        budget_used=BudgetSnapshot(),
        child_state=None,
    )
    r = await h.resume(state, HumanResponseAllow())
    assert isinstance(r, RunResultSuccess)
    assert r.output == "done"
    assert reg.call_count == 1


# ---------------------------------------------------------------------------
# Rule: non-ReAct strategies are explicitly marked NotYetImplemented.
# ---------------------------------------------------------------------------


async def test_non_react_strategies_marked_not_yet_implemented() -> None:
    a = _agent()
    h = StandardHarness(_config(a))
    strategies = [
        LoopStrategyRalph(),
        LoopStrategySelfVerifying(),
        LoopStrategyPlanExecute(plan_model=None),
        LoopStrategyHillClimbing(
            direction="maximize",
            max_stagnation=None,
            revert_on_no_improvement=False,
            min_improvement_delta=None,
        ),
    ]
    for s in strategies:
        t = Task.new("do", SessionId("s1"), s)
        r = await h.run(HarnessRunOptions(t))
        assert isinstance(r, RunResultFailure)
        assert isinstance(r.reason, HaltReasonStrategyNotYetImplemented)


# ---------------------------------------------------------------------------
# Rule: Aggregate usage accumulates across turns.
# ---------------------------------------------------------------------------


async def test_aggregate_usage_accumulates() -> None:
    a = _agent()
    a.push(
        ToolCallRequested(
            calls=[_tc()],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
    )
    a.push(
        FinalResponse(
            content="ok",
            usage=TokenUsage(input_tokens=7, output_tokens=3),
        )
    )
    reg = ScriptedToolRegistry().push(ToolOutputSuccess(content="x"))
    h = StandardHarness(_config(a, tool_registry=reg))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.usage.input_tokens == 17
    assert r.usage.output_tokens == 8


# ---------------------------------------------------------------------------
# Serde round-trip — fixture portability.
# ---------------------------------------------------------------------------


def test_run_result_roundtrips_json() -> None:
    r = RunResultFailure(
        reason=HaltReasonBudgetExceeded(limit_type="turns"),
        session_id=SessionId("s"),
        usage=AggregateUsage(),
        turns=3,
    )
    adapter = TypeAdapter(RunResult)
    s = adapter.dump_json(r)
    back = adapter.validate_json(s)
    assert back == r


def test_paused_state_roundtrips_json() -> None:
    ps = PausedState(
        session_id=SessionId("s"),
        task_id=TaskId("t"),
        turn_number=4,
        session_state=SessionState(),
        pending_tool_calls=[_tc()],
        approved_results=[],
        human_request=HumanRequestClarification(question="what?"),
        task=_react_task(),
        budget_used=BudgetSnapshot(
            turns=4, input_tokens=100, output_tokens=50, wall_time=10, cost_usd=0.0
        ),
        child_state=None,
    )
    s = ps.model_dump_json()
    back = PausedState.model_validate_json(s)
    assert back == ps


def test_child_paused_state_has_no_child_field() -> None:
    """ChildPausedState cannot nest itself — spec depth-1 rule."""
    cs = ChildPausedState(
        session_id=SessionId("c"),
        task_id=TaskId("ct"),
        turn_number=1,
        session_state=SessionState(),
        pending_tool_calls=[],
        approved_results=[],
        human_request=HumanRequestClarification(question="?"),
        task=_react_task(1),
        budget_used=BudgetSnapshot(),
        parent_tool_call_id="p",
    )
    j = cs.model_dump_json()
    assert "child_state" not in j
    # Also confirm the type literally has no such field.
    assert "child_state" not in ChildPausedState.model_fields


# ---------------------------------------------------------------------------
# Fixture-replay integration test — mirrors
# rust/crates/spore-core/tests/harness_fixture_replay.rs
# ---------------------------------------------------------------------------


def _fixture_path() -> Path:
    # tests/  →  spore_core/  →  packages/  →  python/  →  repo-root
    here = Path(__file__).resolve()
    return here.parents[4] / "fixtures" / "model_responses" / "harness" / "react_loop.jsonl"


async def test_react_loop_dispatches_tool_then_completes() -> None:
    """Replays the shared fixture and asserts the same outcome as the Rust
    integration test."""
    jsonl = _fixture_path().read_text()
    replay = ReplayModelInterface.from_jsonl(
        jsonl,
        ProviderInfo(name="anthropic", model_id="fixture", context_window=200_000),
    )
    agent = ModelAgent(AgentId("fixture-agent"), replay)

    reg = ScriptedToolRegistry().push(ToolOutputSuccess(content="127.0.0.1 localhost"))
    config = HarnessConfig(
        agent=agent,
        tool_registry=reg,
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
    )
    harness = StandardHarness(config)
    task = Task.new(
        "read /etc/hosts then summarize",
        SessionId("fixture-session"),
        LoopStrategyReAct(max_iterations=5),
    )
    r = await harness.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "127.0.0.1 localhost"
    assert r.turns == 2, "ReAct loop should run two turns"
    assert r.usage.input_tokens == 30, "12 + 18 input tokens"
    assert r.usage.output_tokens == 14, "8 + 6 output tokens"
    assert reg.call_count == 1
