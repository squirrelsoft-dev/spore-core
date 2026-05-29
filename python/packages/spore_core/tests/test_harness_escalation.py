"""Tests for the Tool Escalation Protocol (issue #80).

Mirrors ``rust/crates/spore-core/src/harness.rs`` escalation unit tests and the
shared fixture-replay integration tests. Each test exercises one rule (R1–R9);
the rule lives in the test docstring.

Rules:

* R1 — a dispatched ``Escalate`` terminates the run and returns
  :class:`RunResultEscalate` (not :class:`RunResultFailure`).
* R2 — the escalation is NOT appended to message history (it is a control
  signal, not a conversation turn).
* R3 — observability is finalized with :class:`SessionOutcomeEscalated`, in
  contrast with a ``WaitingForHuman`` pause, which is never finalized.
* R4 — all five fields of :class:`RunResultEscalate` (``signal``, ``state``,
  ``session_id``, ``usage``, ``turns``) are populated.
* R5 — the preserved ``state`` is a resumable :class:`PausedState` with
  ``human_request = None``.
* R6 — the signal is discarded on resume (the harness never re-acts on it).
* R7 — every :class:`HarnessSignal` variant + the wrapping
  :class:`ToolOutputEscalate` / :class:`RunResultEscalate` round-trip through
  the JSON wire format byte-identically (via the shared fixture).
* R8 — ``Abort`` surfaces as :class:`RunResultEscalate`, NOT
  :class:`RunResultFailure`.
* R9 — remaining tool calls in the batch are preserved into
  ``pending_tool_calls``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetSnapshot,
    HarnessConfig,
    HarnessRunOptions,
    HarnessSignal,
    HarnessSignalAbort,
    HarnessSignalEnterPlanMode,
    HarnessSignalExitPlanMode,
    HarnessSignalSwitchMode,
    LoopStrategyReAct,
    MockAgent,
    NoopContextManager,
    PausedState,
    ProviderInfo,
    ReplayModelInterface,
    RunResult,
    RunResultEscalate,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    StandardHarness,
    Task,
    ToolCall,
    ToolCallRequested,
    ToolOutput,
    ToolOutputEscalate,
    ToolOutputSuccess,
    TokenUsage,
)
from spore_core.agent import Context, ModelAgent
from spore_core.harness import HarnessToolResult
from spore_core.hooks import PlanArtifact
from spore_core.model import Message, ModelParams
from spore_core.prompt_chunk_registry import Mode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage(in_t: int = 1, out_t: int = 1) -> TokenUsage:
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


def _tc(call_id: str = "c", name: str = "x") -> ToolCall:
    return ToolCall(id=call_id, name=name, input={})


def _react_task(session: str = "s1", max_iter: int = 5) -> Task:
    return Task.new(
        "investigate then escalate",
        SessionId(session),
        LoopStrategyReAct(max_iterations=max_iter),
    )


def _abort_signal() -> HarnessSignalAbort:
    return HarnessSignalAbort(reason="user asked to stop")


class RecordingContextManager:
    """Records every appended message so a test can assert what reached
    history. Satisfies the :class:`ContextManager` protocol structurally."""

    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.tool_results: list[HarnessToolResult] = []

    async def assemble(self, session: SessionState, task: Task) -> Context:
        _ = task
        return Context(messages=list(self.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: HarnessToolResult) -> None:
        _ = session
        self.tool_results.append(result)

    async def append_assistant_message(self, session: SessionState, message: Message) -> None:
        _ = session
        self.messages.append(message)

    async def append_user_message(self, session: SessionState, text: str) -> None:
        _ = session

    def should_compact(self, session: SessionState) -> bool:
        _ = session
        return False


class RecordingObservability:
    """Captures the terminal outcome handed to ``set_session_outcome`` and the
    set of sessions that were flushed, so a test can assert the escalation
    finalize path. Satisfies the :class:`ObservabilityProvider` protocol
    structurally (all span emitters are no-ops)."""

    def __init__(self) -> None:
        self.outcomes: dict[str, object] = {}
        self.flushed: list[str] = []

    def emit_turn(self, span: object) -> None:  # noqa: D401
        _ = span

    def emit_tool_call(self, span: object) -> None:
        _ = span

    def emit_sensor(self, span: object) -> None:
        _ = span

    def emit_context(self, span: object) -> None:
        _ = span

    def emit_middleware(self, span: object) -> None:
        _ = span

    def emit_warn(self, span: object) -> None:
        _ = span

    def set_session_outcome(self, session_id: SessionId, outcome: object) -> None:
        self.outcomes[str(session_id)] = outcome

    async def flush_session(self, session_id: SessionId) -> None:
        self.flushed.append(str(session_id))


def _escalating_config(
    *,
    signal: HarnessSignal,
    tool_registry: ScriptedToolRegistry | None = None,
    context_manager: object | None = None,
    observability: object | None = None,
) -> HarnessConfig:
    reg = tool_registry or ScriptedToolRegistry().push(ToolOutputEscalate(signal=signal))
    return HarnessConfig(
        agent=MockAgent(AgentId("esc")),
        tool_registry=reg,
        sandbox=AllowAllSandbox(),
        context_manager=context_manager or NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        observability=observability,
    )


# ---------------------------------------------------------------------------
# R1 + R8: a dispatched Escalate { Abort } terminates the run and returns
# RunResultEscalate, NOT RunResultFailure.
# ---------------------------------------------------------------------------


async def test_escalate_abort_returns_escalate_not_failure() -> None:
    """R1/R8: ``Abort`` is an intentional, clean stop — it surfaces as
    :class:`RunResultEscalate`, never :class:`RunResultFailure`."""
    agent = MockAgent(AgentId("esc"))
    agent.push(ToolCallRequested(calls=[_tc("c1", "abort_tool")], usage=_usage()))

    config = _escalating_config(signal=_abort_signal())
    config.agent = agent
    harness = StandardHarness(config)

    r = await harness.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultEscalate)
    assert isinstance(r.signal, HarnessSignalAbort)
    assert r.signal.reason == "user asked to stop"


# ---------------------------------------------------------------------------
# R2: the escalation is NOT appended to message history.
# ---------------------------------------------------------------------------


async def test_escalation_not_appended_to_history() -> None:
    """R2: the escalating tool's result must NOT be appended as a tool result —
    the assistant tool-call turn is recorded (as for any turn), but no tool
    result for the escalating call reaches history."""
    cm = RecordingContextManager()
    agent = MockAgent(AgentId("esc"))
    agent.push(ToolCallRequested(calls=[_tc("c1", "abort_tool")], usage=_usage()))

    config = _escalating_config(signal=_abort_signal(), context_manager=cm)
    config.agent = agent
    harness = StandardHarness(config)

    r = await harness.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultEscalate)
    # No tool result was appended for the escalating call.
    assert cm.tool_results == []
    # And no message records the escalation result content.
    assert all("escalat" not in str(m.content).lower() for m in cm.messages)


# ---------------------------------------------------------------------------
# R3: observability is finalized with SessionOutcomeEscalated.
# ---------------------------------------------------------------------------


async def test_escalation_finalizes_observability_as_escalated() -> None:
    """R3: a terminal escalation finalizes observability with
    :class:`SessionOutcomeEscalated` and flushes the session — distinct from a
    ``WaitingForHuman`` pause, which is never finalized."""
    from spore_core import SessionOutcomeEscalated

    obs = RecordingObservability()
    agent = MockAgent(AgentId("esc"))
    agent.push(ToolCallRequested(calls=[_tc("c1", "abort_tool")], usage=_usage()))

    config = _escalating_config(signal=_abort_signal(), observability=obs)
    config.agent = agent
    harness = StandardHarness(config)

    r = await harness.run(HarnessRunOptions(_react_task("esc-session")))
    assert isinstance(r, RunResultEscalate)
    assert "esc-session" in obs.flushed
    assert isinstance(obs.outcomes["esc-session"], SessionOutcomeEscalated)


async def test_waiting_for_human_not_finalized_contrast() -> None:
    """R3 contrast: a ``WaitingForHuman`` pause is NOT terminal — observability
    is neither outcome-stamped nor flushed."""
    from spore_core import HumanRequestClarification, RunResultWaitingForHuman
    from spore_core.harness import ChildPausedState, ToolOutputWaitingForHuman

    obs = RecordingObservability()
    agent = MockAgent(AgentId("hitl"))
    agent.push(ToolCallRequested(calls=[_tc("c1", "ask_tool")], usage=_usage()))

    child = ChildPausedState(
        session_id=SessionId("child"),
        task_id=Task.new("c", SessionId("child"), LoopStrategyReAct(max_iterations=1)).id,
        turn_number=0,
        session_state=SessionState(),
        task=Task.new("c", SessionId("child"), LoopStrategyReAct(max_iterations=1)),
        budget_used=BudgetSnapshot(),
        parent_tool_call_id="c1",
        human_request=HumanRequestClarification(question="?"),
    )
    reg = ScriptedToolRegistry().push(
        ToolOutputWaitingForHuman(
            child_state=child, request=HumanRequestClarification(question="?")
        )
    )
    config = _escalating_config(signal=_abort_signal(), tool_registry=reg, observability=obs)
    config.agent = agent
    harness = StandardHarness(config)

    r = await harness.run(HarnessRunOptions(_react_task("hitl-session")))
    assert isinstance(r, RunResultWaitingForHuman)
    assert obs.flushed == []
    assert obs.outcomes == {}


# ---------------------------------------------------------------------------
# R4: all five RunResultEscalate fields are populated.
# ---------------------------------------------------------------------------


async def test_escalate_populates_all_five_fields() -> None:
    """R4: ``signal`` + ``state`` + ``session_id`` + ``usage`` + ``turns`` are
    all present and consistent with the run accounting."""
    agent = MockAgent(AgentId("esc"))
    agent.push(ToolCallRequested(calls=[_tc("c1", "abort_tool")], usage=_usage(7, 3)))

    config = _escalating_config(signal=_abort_signal())
    config.agent = agent
    harness = StandardHarness(config)

    r = await harness.run(HarnessRunOptions(_react_task("five-session")))
    assert isinstance(r, RunResultEscalate)
    assert isinstance(r.signal, HarnessSignalAbort)
    assert isinstance(r.state, PausedState)
    assert str(r.session_id) == "five-session"
    assert r.usage.input_tokens == 7
    assert r.usage.output_tokens == 3
    assert r.turns == 1


# ---------------------------------------------------------------------------
# R5: the preserved state is a resumable PausedState with human_request = None.
# ---------------------------------------------------------------------------


async def test_escalate_state_is_resumable_with_no_human_request() -> None:
    """R5: ``RunResultEscalate.state`` is a full :class:`PausedState` carrying
    ``human_request = None`` (an escalation has no human request)."""
    agent = MockAgent(AgentId("esc"))
    agent.push(ToolCallRequested(calls=[_tc("c1", "abort_tool")], usage=_usage()))

    config = _escalating_config(signal=_abort_signal())
    config.agent = agent
    harness = StandardHarness(config)

    r = await harness.run(HarnessRunOptions(_react_task("resume-session")))
    assert isinstance(r, RunResultEscalate)
    assert r.state.human_request is None
    assert str(r.state.session_id) == "resume-session"
    assert r.state.child_state is None
    # human_request serializes as an explicit null (matches the Rust
    # #[serde(default)] wire shape — field present, value null).
    dumped = r.state.model_dump(mode="json")
    assert "human_request" in dumped
    assert dumped["human_request"] is None


# ---------------------------------------------------------------------------
# R6: the signal is discarded on resume (the harness never re-acts on it).
# ---------------------------------------------------------------------------


async def test_signal_discarded_on_resume() -> None:
    """R6: resuming the preserved state re-enters the loop; the signal is NOT
    stored in :class:`PausedState`, so the harness never re-acts on it. With no
    pending calls and a final response queued, the resume reaches a clean
    terminal outcome with no trace of the escalation."""
    from spore_core import FinalResponse, HumanResponseAllow

    agent = MockAgent(AgentId("esc"))
    agent.push(ToolCallRequested(calls=[_tc("c1", "abort_tool")], usage=_usage()))
    # The escalating call is the only call in the batch, so resume has no
    # pending calls; the next queued turn is a plain final response.
    agent.push(FinalResponse(content="resumed cleanly", usage=_usage()))

    config = _escalating_config(signal=_abort_signal())
    config.agent = agent
    harness = StandardHarness(config)

    r = await harness.run(HarnessRunOptions(_react_task("disc-session")))
    assert isinstance(r, RunResultEscalate)
    # PausedState carries no signal field at all.
    assert "signal" not in r.state.model_dump(mode="json")

    resumed = await harness.resume(r.state, HumanResponseAllow())
    assert isinstance(resumed, RunResultSuccess)
    assert resumed.output == "resumed cleanly"


# ---------------------------------------------------------------------------
# R9: remaining tool calls in the batch are preserved into pending_tool_calls.
# ---------------------------------------------------------------------------


async def test_remaining_calls_preserved_in_pending() -> None:
    """R9: when an early call in a batch escalates, the remaining calls are
    preserved into ``pending_tool_calls`` (mirroring ``WaitingForHuman``); the
    earlier-dispatched calls are kept in ``approved_results``."""
    agent = MockAgent(AgentId("esc"))
    agent.push(
        ToolCallRequested(
            calls=[_tc("c0", "ok_tool"), _tc("c1", "abort_tool"), _tc("c2", "later_tool")],
            usage=_usage(),
        )
    )
    # First call succeeds, second escalates; third must never dispatch.
    reg = (
        ScriptedToolRegistry()
        .push(ToolOutputSuccess(content="ok"))
        .push(ToolOutputEscalate(signal=_abort_signal()))
        .push(ToolOutputSuccess(content="should-not-run"))
    )
    config = _escalating_config(signal=_abort_signal(), tool_registry=reg)
    config.agent = agent
    harness = StandardHarness(config)

    r = await harness.run(HarnessRunOptions(_react_task("pending-session")))
    assert isinstance(r, RunResultEscalate)
    # Only the first two calls dispatched; the third was preserved, not run.
    assert reg.call_count == 2
    pending_ids = [c.id for c in r.state.pending_tool_calls]
    assert pending_ids == ["c2"]
    approved_ids = [tr.call_id for tr in r.state.approved_results]
    assert approved_ids == ["c0"]


# ===========================================================================
# R7: wire round-trip — shared serde fixture (all four HarnessSignal variants
# + RunResultEscalate cases). Mirrors the Rust serde round-trip fixture test.
# ===========================================================================


def _fixtures_root() -> Path:
    # tests/ → spore_core/ → packages/ → python/ → repo-root
    return Path(__file__).resolve().parents[4] / "fixtures"


def _signals_fixture() -> Path:
    return _fixtures_root() / "harness" / "escalation_signals.json"


_TOOL_OUTPUT_ADAPTER = TypeAdapter(ToolOutput)
_RUN_RESULT_ADAPTER = TypeAdapter(RunResult)


def test_serde_round_trip_tool_output_cases() -> None:
    """R7: every ``ToolOutput::Escalate`` case in the shared fixture (all four
    ``HarnessSignal`` variants) deserializes to a :class:`ToolOutputEscalate`
    and re-serializes byte-identically to the canonical JSON."""
    path = _signals_fixture()
    if not path.exists():
        pytest.skip("shared escalation fixture not present")
    doc = json.loads(path.read_text())

    kinds_seen: set[str] = set()
    for case in doc["tool_output_cases"]:
        parsed = _TOOL_OUTPUT_ADAPTER.validate_python(case)
        assert isinstance(parsed, ToolOutputEscalate)
        kinds_seen.add(parsed.signal.kind)
        # Re-serialize and compare to the canonical fixture object.
        round_tripped = parsed.model_dump(mode="json")
        assert round_tripped == case

    assert kinds_seen == {"enter_plan_mode", "exit_plan_mode", "switch_mode", "abort"}


def test_serde_round_trip_run_result_cases() -> None:
    """R7: every ``RunResult::Escalate`` case in the shared fixture deserializes
    to a :class:`RunResultEscalate` (with ``state.human_request = None``) and
    re-serializes byte-identically."""
    path = _signals_fixture()
    if not path.exists():
        pytest.skip("shared escalation fixture not present")
    doc = json.loads(path.read_text())

    for case in doc["run_result_cases"]:
        parsed = _RUN_RESULT_ADAPTER.validate_python(case)
        assert isinstance(parsed, RunResultEscalate)
        assert parsed.state.human_request is None
        round_tripped = parsed.model_dump(mode="json")
        assert round_tripped == case


def test_signal_variants_construct_from_fixture_payloads() -> None:
    """R7: the concrete signal classes accept the fixture payloads directly and
    preserve the carried data (PlanArtifact / Mode / context / reason)."""
    enter = HarnessSignalEnterPlanMode(context="investigated; ready to plan")
    assert enter.model_dump(mode="json") == {
        "kind": "enter_plan_mode",
        "context": "investigated; ready to plan",
    }

    plan = PlanArtifact(tasks=["scaffold", "test"], rationale="incrementally")
    exit_ = HarnessSignalExitPlanMode(plan=plan)
    assert exit_.model_dump(mode="json")["kind"] == "exit_plan_mode"
    assert exit_.plan.tasks == ["scaffold", "test"]

    switch = HarnessSignalSwitchMode(mode=Mode.PLAN)
    assert switch.model_dump(mode="json") == {"kind": "switch_mode", "mode": "plan"}

    abort = HarnessSignalAbort(reason="stop")
    assert abort.model_dump(mode="json") == {"kind": "abort", "reason": "stop"}


# ===========================================================================
# Fixture-replay: drive the loop from the shared model-response fixture and
# assert the harness returns RunResultEscalate. Mirrors the Rust loop-replay
# integration test.
# ===========================================================================


def _loop_fixture() -> Path:
    return _fixtures_root() / "model_responses" / "harness" / "escalation_loop.jsonl"


async def test_escalation_loop_replay_returns_escalate() -> None:
    """Replays the shared ``escalation_loop.jsonl`` fixture: the model emits a
    single ``abort_tool`` call; the tool registry maps it to an ``Abort``
    escalation. The harness must return :class:`RunResultEscalate` carrying the
    ``Abort`` signal (NOT a failure), and the escalation must not be appended to
    history."""
    path = _loop_fixture()
    if not path.exists():
        pytest.skip("shared escalation loop fixture not present")
    jsonl = path.read_text()
    replay = ReplayModelInterface.from_jsonl(
        jsonl,
        ProviderInfo(name="anthropic", model_id="fixture", context_window=200_000),
    )
    agent = ModelAgent(AgentId("fixture-agent"), replay)

    cm = RecordingContextManager()
    reg = ScriptedToolRegistry().push(
        ToolOutputEscalate(signal=HarnessSignalAbort(reason="blocked on missing credentials"))
    )
    config = HarnessConfig(
        agent=agent,
        tool_registry=reg,
        sandbox=AllowAllSandbox(),
        context_manager=cm,
        termination_policy=AlwaysContinuePolicy(),
    )
    harness = StandardHarness(config)
    task = Task.new(
        "investigate then decide whether to abort",
        SessionId("fixture-escalation"),
        LoopStrategyReAct(max_iterations=5),
    )

    r = await harness.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultEscalate)
    assert isinstance(r.signal, HarnessSignalAbort)
    assert r.signal.reason == "blocked on missing credentials"
    assert reg.call_count == 1
    # R2 on the replay path: no tool result appended for the escalating call.
    assert cm.tool_results == []
