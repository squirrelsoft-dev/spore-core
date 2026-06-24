"""Tests for the mid-loop consult primitive (issue #114) — worker + resume seams.

Mirrors ``rust/crates/spore-core/src/harness.rs`` consult unit tests and the
shared fixture-replay test. Each test exercises one rule (R1, R5b, R6, R8, R9,
R10); the orchestrator-mediation rules (R2/R3/R4/R5a/R7) live in
``packages/spore_tools/tests/test_subagent_consult.py``.

Rules:

* R1 — a worker-side ``ToolOutputConsult`` pauses the loop and returns
  :class:`RunResultConsult`; the consult call is the HEAD of
  ``pending_tool_calls``, ``human_request`` is ``None``, ``child_state`` is
  ``None``.
* R6 — graceful degradation: an empty ``consult_handlers`` map means a
  standalone worker surfaces :class:`RunResultConsult` unchanged.
* R8 — every consult type round-trips through the shared
  ``fixtures/harness/consult.json`` byte-identically.
* R9 — existing callers (no handlers) are unaffected: the default config has an
  empty map and the consult surfaces unchanged.
* R10 — the consult is NOT appended to message history until the resume injects
  the answer as the head call's tool RESULT; ``resume_consult`` then continues.
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
    ConsultOverflowPolicy,
    ConsultOverflowPolicyEscalateToHuman,
    ConsultOverflowPolicySoftFail,
    ConsultRequest,
    ConsultResponse,
    ConsultResponseAnswer,
    ConsultResponseBudgetExhausted,
    HarnessConfig,
    HarnessRunOptions,
    ReactConfig,
    MockAgent,
    NoopContextManager,
    PausedState,
    RunResult,
    RunResultConsult,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    StandardHarness,
    Task,
    TaskId,
    ToolCall,
    ToolOutput,
    ToolOutputConsult,
    TokenUsage,
)
from spore_core.agent import FinalResponse, ToolCallRequested
from spore_core.harness import HarnessToolResult
from spore_core.model import Message, ModelParams
from spore_core.observability import (
    ContextOperationConsultResumed,
    ContextOperationConsultSpawned,
    InMemoryObservabilityProvider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage(in_t: int = 1, out_t: int = 1) -> TokenUsage:
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


def _react_task(session: str = "s1", max_iter: int = 5) -> Task:
    return Task.new("audit the auth module", SessionId(session), ReactConfig.per_loop(max_iter))


def _consult_req() -> ConsultRequest:
    return ConsultRequest(
        kind="advice",
        situation="stuck on auth",
        attempts=3,
        question="what next?",
    )


class RecordingContextManager:
    """Records appended messages / tool results so a test can assert what reached
    history (R10). Satisfies the ``ContextManager`` Protocol structurally."""

    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.tool_results: list[HarnessToolResult] = []

    async def assemble(self, session: SessionState, task: Task, sources: object) -> object:
        from spore_core.agent import Context

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


def _config(
    agent: MockAgent,
    *,
    tool_registry: ScriptedToolRegistry | None = None,
    context_manager: object | None = None,
    observability: object | None = None,
) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=tool_registry or ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=context_manager or NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        observability=observability,
    )


# ---------------------------------------------------------------------------
# R1 + R10: a worker-side ToolOutputConsult pauses and returns RunResultConsult.
# ---------------------------------------------------------------------------


async def test_worker_consult_pauses_and_returns_run_result_consult() -> None:
    """R1/R10: the consult call is the HEAD of ``pending_tool_calls``,
    ``human_request`` / ``child_state`` are ``None``, the remaining batched call
    is preserved (not dispatched), and no tool result is appended yet (R10)."""
    agent = MockAgent(AgentId("worker"))
    agent.push(
        ToolCallRequested(
            calls=[
                ToolCall(id="c0", name="ask_advice", input={"kind": "advice"}),
                ToolCall(id="c1", name="x", input={}),
            ],
            usage=_usage(),
        )
    )
    cm = RecordingContextManager()
    reg = ScriptedToolRegistry().push(ToolOutputConsult.consult(_consult_req()))
    h = StandardHarness(_config(agent, tool_registry=reg, context_manager=cm))

    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultConsult)
    assert r.request.kind == "advice"
    assert r.request.question == "what next?"
    # R1: human_request None, no child_state.
    assert r.state.human_request is None
    assert r.state.child_state is None
    # R1: consulting call (c0) is head; c1 trails.
    assert [c.id for c in r.state.pending_tool_calls] == ["c0", "c1"]
    # R10: no tool result was appended (the consult is not a conversation turn).
    assert cm.tool_results == []
    # R1/R9: only the consulting call was dispatched; c1 was preserved, not run.
    assert reg.call_count == 1


# ---------------------------------------------------------------------------
# R6 + R9: an empty consult_handlers map surfaces RunResultConsult unchanged.
# ---------------------------------------------------------------------------


async def test_empty_consult_handlers_surfaces_consult_unchanged() -> None:
    """R6/R9: the default config has an empty ``consult_handlers`` map, so a
    standalone worker simply surfaces :class:`RunResultConsult` to its caller —
    existing callers without a handler map are unaffected."""
    agent = MockAgent(AgentId("worker"))
    agent.push(
        ToolCallRequested(
            calls=[ToolCall(id="c0", name="ask_advice", input={"kind": "advice"})], usage=_usage()
        )
    )
    config = _config(
        agent, tool_registry=ScriptedToolRegistry().push(ToolOutputConsult.consult(_consult_req()))
    )
    assert config.consult_handlers == {}
    h = StandardHarness(config)
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultConsult)


# ---------------------------------------------------------------------------
# R10 + resume seam: resume_consult injects the answer as the head call's tool
# RESULT, then continues to Success.
# ---------------------------------------------------------------------------


async def test_resume_consult_injects_answer_as_tool_result() -> None:
    """R10: ``resume_consult`` with an :class:`ConsultResponseAnswer` injects the
    text as the tool RESULT for the head pending (consult) call, then continues
    the ReAct loop to Success."""
    agent = MockAgent(AgentId("worker"))
    agent.push(FinalResponse(content="consult-resumed-done", usage=_usage()))
    cm = RecordingContextManager()
    h = StandardHarness(_config(agent, context_manager=cm))
    state = PausedState(
        session_id=SessionId("s"),
        task_id=TaskId("t"),
        turn_number=1,
        session_state=SessionState(),
        pending_tool_calls=[ToolCall(id="consult", name="ask_advice", input={"kind": "advice"})],
        approved_results=[],
        human_request=None,
        task=_react_task("s"),
        budget_used=BudgetSnapshot(),
        child_state=None,
    )
    r = await h.resume_consult(state, ConsultResponseAnswer(text="the answer"))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "consult-resumed-done"
    # The answer reached history exactly once, as the consult call's tool result.
    assert len(cm.tool_results) == 1
    tr = cm.tool_results[0]
    assert tr.call_id == "consult"
    assert tr.output.kind == "success"
    assert tr.output.content == "the answer"


async def test_resume_consult_budget_exhausted_also_resumes() -> None:
    """R5a (worker side): a ``BudgetExhausted`` resume injects its message as the
    tool result and continues — the worker finishes with what it has."""
    agent = MockAgent(AgentId("worker"))
    agent.push(FinalResponse(content="finished-without-help", usage=_usage()))
    cm = RecordingContextManager()
    h = StandardHarness(_config(agent, context_manager=cm))
    state = PausedState(
        session_id=SessionId("s"),
        task_id=TaskId("t"),
        turn_number=1,
        session_state=SessionState(),
        pending_tool_calls=[ToolCall(id="consult", name="ask_advice", input={"kind": "advice"})],
        approved_results=[],
        human_request=None,
        task=_react_task("s"),
        budget_used=BudgetSnapshot(),
        child_state=None,
    )
    r = await h.resume_consult(state, ConsultResponseBudgetExhausted(message="budget gone"))
    assert isinstance(r, RunResultSuccess)
    assert cm.tool_results[0].output.content == "budget gone"


# ---------------------------------------------------------------------------
# Observability: lightweight consult-spawned / consult-resumed events alongside
# the existing SkillInjected event.
# ---------------------------------------------------------------------------


async def test_consult_emits_spawn_event() -> None:
    """A worker pause emits a :class:`ContextOperationConsultSpawned` context
    span carrying the consult kind."""
    agent = MockAgent(AgentId("worker"))
    agent.push(
        ToolCallRequested(
            calls=[ToolCall(id="c0", name="ask_advice", input={"kind": "advice"})], usage=_usage()
        )
    )
    obs = InMemoryObservabilityProvider()
    reg = ScriptedToolRegistry().push(ToolOutputConsult.consult(_consult_req()))
    h = StandardHarness(_config(agent, tool_registry=reg, observability=obs))

    r = await h.run(HarnessRunOptions(_react_task("spawn-session")))
    assert isinstance(r, RunResultConsult)
    spawned = [
        c.operation
        for c in obs.context_spans(SessionId("spawn-session"))
        if isinstance(c.operation, ContextOperationConsultSpawned)
    ]
    assert len(spawned) == 1
    assert spawned[0].consult_kind == "advice"


async def test_resume_consult_emits_resumed_event() -> None:
    """``resume_consult`` emits a :class:`ContextOperationConsultResumed` span;
    an Answer sets ``answered=True``, a BudgetExhausted sets ``answered=False``."""
    obs = InMemoryObservabilityProvider()
    agent = MockAgent(AgentId("worker"))
    agent.push(FinalResponse(content="done", usage=_usage()))
    h = StandardHarness(_config(agent, observability=obs))
    state = PausedState(
        session_id=SessionId("resume-session"),
        task_id=TaskId("t"),
        turn_number=1,
        session_state=SessionState(),
        pending_tool_calls=[ToolCall(id="consult", name="ask_advice", input={"kind": "advice"})],
        approved_results=[],
        human_request=None,
        task=_react_task("resume-session"),
        budget_used=BudgetSnapshot(),
        child_state=None,
    )
    await h.resume_consult(state, ConsultResponseAnswer(text="ans"))
    resumed = [
        c.operation
        for c in obs.context_spans(SessionId("resume-session"))
        if isinstance(c.operation, ContextOperationConsultResumed)
    ]
    assert len(resumed) == 1
    assert resumed[0].consult_kind == "advice"
    assert resumed[0].answered is True


# ===========================================================================
# R8: wire round-trip — shared serde fixture (fixtures/harness/consult.json).
# ===========================================================================


def _fixtures_root() -> Path:
    # tests/ → spore_core/ → packages/ → python/ → repo-root
    return Path(__file__).resolve().parents[4] / "fixtures"


def _consult_fixture() -> Path:
    return _fixtures_root() / "harness" / "consult.json"


_RUN_RESULT_ADAPTER = TypeAdapter(RunResult)
_TOOL_OUTPUT_ADAPTER = TypeAdapter(ToolOutput)
_CONSULT_REQUEST_ADAPTER = TypeAdapter(ConsultRequest)
_CONSULT_RESPONSE_ADAPTER = TypeAdapter(ConsultResponse)
_CONSULT_OVERFLOW_ADAPTER = TypeAdapter(ConsultOverflowPolicy)


def _check(
    doc: dict, key: str, adapter: TypeAdapter, expected_type: type | tuple[type, ...]
) -> None:
    arr = doc[key]
    assert isinstance(arr, list) and arr, f"`{key}` must have cases"
    for i, case in enumerate(arr):
        parsed = adapter.validate_python(case)
        assert isinstance(parsed, expected_type), f"`{key}`[{i}] type"
        back = adapter.dump_python(parsed, mode="json")
        assert back == case, f"`{key}`[{i}] not byte-identical"


def test_consult_fixture_replay_round_trips() -> None:
    """R8: every section of the shared ``consult.json`` deserializes to the typed
    value and re-serializes byte-identically — proving the wire shape is stable
    and the same fixture replays identically across the four languages."""
    path = _consult_fixture()
    if not path.exists():
        pytest.skip("shared consult fixture not present")
    doc = json.loads(path.read_text())

    _check(doc, "run_result_cases", _RUN_RESULT_ADAPTER, RunResultConsult)
    _check(doc, "worker_tool_output_cases", _TOOL_OUTPUT_ADAPTER, ToolOutputConsult)
    _check(doc, "subagent_tool_output_cases", _TOOL_OUTPUT_ADAPTER, ToolOutputConsult)
    _check(doc, "consult_request_cases", _CONSULT_REQUEST_ADAPTER, ConsultRequest)
    _check(
        doc,
        "consult_response_cases",
        _CONSULT_RESPONSE_ADAPTER,
        (ConsultResponseAnswer, ConsultResponseBudgetExhausted),
    )
    _check(
        doc,
        "consult_overflow_policy_cases",
        _CONSULT_OVERFLOW_ADAPTER,
        (ConsultOverflowPolicySoftFail, ConsultOverflowPolicyEscalateToHuman),
    )

    # Spot-check the documented invariants on the structured cases.
    rr = doc["run_result_cases"][0]
    assert rr["kind"] == "consult"
    assert rr["state"]["human_request"] is None
    assert rr["state"]["child_state"] is None
    # Worker-side ToolOutputConsult omits child_state (excluded when None).
    worker = doc["worker_tool_output_cases"][0]
    assert worker["kind"] == "consult"
    assert "child_state" not in worker
    # Subagent-boundary ToolOutputConsult carries a populated child_state.
    sub = doc["subagent_tool_output_cases"][0]
    assert isinstance(sub["child_state"], dict)
