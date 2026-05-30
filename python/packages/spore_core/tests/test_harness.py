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
    HaltReasonHillClimbingMisconfigured,
    HaltReasonHumanHalted,
    HaltReasonMiddlewareHalt,
    HaltReasonSandboxViolation,
    HaltReasonStrategyNotYetImplemented,
    HaltReasonTerminationPolicyHalt,
    HaltReasonUnrecoverableToolError,
    HarnessBuilder,
    HarnessConfig,
    HarnessRunOptions,
    HumanRequestClarification,
    HumanRequestToolApproval,
    HumanResponseAllow,
    HumanResponseHalt,
    LoopStrategyHillClimbing,
    LoopStrategyReAct,
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
from spore_core.agent import Context, ModelAgent
from spore_core.harness import HarnessToolResult
from spore_core.model import (
    Message,
    ModelParams,
    Role,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)


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
    # PlanExecute no longer uses StrategyNotYetImplemented; it runs the full
    # two-phase plan→execute loop (#59). SelfVerifying (#61), Ralph (#58), and
    # HillClimbing (#60) are now all implemented. HillClimbing with no
    # metric_evaluator wired returns the typed HillClimbingMisconfigured halt
    # (Decision 6), NOT StrategyNotYetImplemented (mirrors the Rust reference
    # test at harness.rs:7100). No non-ReAct strategy remains stubbed.
    a = _agent()
    h = StandardHarness(_config(a))
    s = LoopStrategyHillClimbing(
        direction="maximize",
        max_stagnation=None,
        revert_on_no_improvement=False,
        min_improvement_delta=None,
    )
    t = Task.new("do", SessionId("s1"), s)
    r = await h.run(HarnessRunOptions(t))
    assert isinstance(r, RunResultFailure)
    assert not isinstance(r.reason, HaltReasonStrategyNotYetImplemented)
    assert isinstance(r.reason, HaltReasonHillClimbingMisconfigured)
    assert "metric_evaluator" in r.reason.reason


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


# ---------------------------------------------------------------------------
# Rule: the harness emits real spans through the durable outbox and flushes a
# session-summary line on terminal success. Mirrors the Rust
# ``harness_emits_spans_through_outbox_jsonl`` integration test.
# ---------------------------------------------------------------------------


async def test_harness_emits_spans_through_outbox_jsonl(tmp_path: Path) -> None:
    """A tool-call-then-final run must write turn and tool_call spans plus a
    trailing session-summary line to ``{root}/sessions/{sid}/trace.jsonl``,
    sharing one trace_id, and drop a ``.flushed`` marker. SPORE_OTLP_ENDPOINT
    is left unset so the run is hermetic (JSONL only)."""
    import json

    session_id = SessionId("outbox-sess")
    agent = MockAgent(AgentId("outbox-agent"))
    agent.push(ToolCallRequested(calls=[_tc("c1", "x")], usage=_usage()))
    agent.push(FinalResponse(content="done", usage=_usage()))

    harness = (
        HarnessBuilder(
            agent,
            ScriptedToolRegistry(),
            AllowAllSandbox(),
            NoopContextManager(),
            AlwaysContinuePolicy(),
        )
        .with_observability_outbox(tmp_path)
        .build()
    )
    task = Task.new("do work", session_id, LoopStrategyReAct(max_iterations=5))
    r = await harness.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultSuccess)

    trace_path = tmp_path / "sessions" / str(session_id) / "trace.jsonl"
    lines = [json.loads(line) for line in trace_path.read_text().splitlines() if line]

    kinds = [line["kind"] for line in lines]
    assert "turn" in kinds
    assert "tool_call" in kinds

    last = lines[-1]
    assert last["kind"] == "session"
    assert last["attributes"]["outcome"] == "success"
    assert last["attributes"]["total_turns"] == 2

    trace_ids = {line["trace_id"] for line in lines}
    assert len(trace_ids) == 1, "all lines share one trace_id"

    assert (tmp_path / "sessions" / str(session_id) / ".flushed").exists()


# ---------------------------------------------------------------------------
# Assistant-turn recording (regression for lost conversation history, #65)
# ---------------------------------------------------------------------------


class RecordingContextManager:
    """A context manager that records every appended message in a shared list so
    tests can inspect the conversation the loop builds. Records assistant turns
    (tool calls + final text) and tool results in order. Mirrors the Rust
    reference ``RecordingContextManager``."""

    def __init__(self) -> None:
        self.messages: list[Message] = []

    async def assemble(self, session: SessionState, task: Task) -> Context:
        _ = task
        return Context(messages=list(session.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: HarnessToolResult) -> None:
        output = result.output
        if isinstance(output, ToolOutputSuccess):
            content, is_error = output.content, False
        elif isinstance(output, ToolOutputError):
            content, is_error = output.message, True
        else:
            content, is_error = "", False
        # Preserve the ``call_id`` on the recorded tool result so the assertion
        # can match an assistant tool-call to its result by id (SPEC QUESTION 1):
        # the recorded shape carries ``tool_use_id`` even though the production
        # adapter's ``append_tool_result`` uses a role:"tool" text content.
        msg = Message(
            role=Role.TOOL,
            content=ToolResultContent(
                tool_use_id=result.call_id, content=content, is_error=is_error
            ),
        )
        session.messages.append(msg)
        self.messages.append(msg)

    async def append_assistant_message(self, session: SessionState, message: Message) -> None:
        session.messages.append(message)
        self.messages.append(message)

    async def append_user_message(self, session: SessionState, text: str) -> None:
        msg = Message(role=Role.USER, content=TextContent(text=text))
        session.messages.append(msg)
        self.messages.append(msg)

    def should_compact(self, session: SessionState) -> bool:
        _ = session
        return False


def _assistant_tool_call_idx(messages: list[Message], call_id: str) -> int | None:
    for i, m in enumerate(messages):
        if (
            m.role == Role.ASSISTANT
            and isinstance(m.content, ToolCallContent)
            and m.content.id == call_id
        ):
            return i
    return None


def _tool_result_idx(messages: list[Message], call_id: str) -> int | None:
    for i, m in enumerate(messages):
        if (
            m.role == Role.TOOL
            and isinstance(m.content, ToolResultContent)
            and m.content.tool_use_id == call_id
        ):
            return i
    return None


async def test_tool_call_records_assistant_message_before_result() -> None:
    """A turn that requests a tool call must record the assistant's tool-call
    message in history, positioned BEFORE the tool result, so the next turn's
    assembled context reflects what the agent already did."""
    a = _agent()
    a.push(ToolCallRequested(calls=[_tc("c1", "read_file")], usage=_usage()))
    a.push(FinalResponse(content="done", usage=_usage()))
    cm = RecordingContextManager()
    reg = ScriptedToolRegistry().push(ToolOutputSuccess(content="contents"))
    h = StandardHarness(_config(a, context_manager=cm, tool_registry=reg))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultSuccess)

    assistant_idx = _assistant_tool_call_idx(cm.messages, "c1")
    tool_idx = _tool_result_idx(cm.messages, "c1")
    assert assistant_idx is not None, "assistant tool-call message must be recorded"
    assert tool_idx is not None, "tool result must be recorded"
    assert assistant_idx < tool_idx, (
        f"assistant tool_use (idx {assistant_idx}) must precede its tool result (idx {tool_idx})"
    )


async def test_final_response_records_assistant_text() -> None:
    """A final response must append the assistant's text to history so a
    continued session sees what the agent said."""
    a = _agent()
    a.push(FinalResponse(content="the final answer", usage=_usage()))
    cm = RecordingContextManager()
    h = StandardHarness(_config(a, context_manager=cm))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultSuccess)

    assert any(
        m.role == Role.ASSISTANT
        and isinstance(m.content, TextContent)
        and m.content.text == "the final answer"
        for m in cm.messages
    ), "assistant final text must be recorded in history"


async def test_resume_after_surface_to_human_records_assistant_once_before_result() -> None:
    """When a run pauses at BeforeTool (SurfaceToHuman) and is resumed with
    Allow, the assistant tool-call message must already be in history — recorded
    before the pause — and positioned before its tool result, with no duplicate
    from the resume path."""
    a = _agent()
    calls = [_tc("c1", "read_file")]
    a.push(ToolCallRequested(calls=calls, usage=_usage()))
    a.push(FinalResponse(content="done", usage=_usage()))
    cm = RecordingContextManager()
    reg = ScriptedToolRegistry().push(ToolOutputSuccess(content="contents"))
    req = HumanRequestToolApproval(calls=calls, risk_level="medium")
    mw = ScriptedMiddleware().push("before_tool", MiddlewareSurfaceToHuman(request=req))
    h = StandardHarness(_config(a, context_manager=cm, tool_registry=reg, middleware=mw))

    paused = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(paused, RunResultWaitingForHuman)

    resumed = await h.resume(paused.state, HumanResponseAllow())
    assert isinstance(resumed, RunResultSuccess)

    assistant_idx = _assistant_tool_call_idx(cm.messages, "c1")
    tool_idx = _tool_result_idx(cm.messages, "c1")
    assert assistant_idx is not None, "assistant tool-call must be recorded on the resume path"
    assert tool_idx is not None, "tool result must be recorded"
    assert assistant_idx < tool_idx, (
        f"assistant tool_use (idx {assistant_idx}) must precede its tool result (idx {tool_idx})"
    )
    count = sum(
        1
        for m in cm.messages
        if m.role == Role.ASSISTANT
        and isinstance(m.content, ToolCallContent)
        and m.content.id == "c1"
    )
    assert count == 1, "assistant tool-call must be recorded exactly once, not duplicated by resume"
