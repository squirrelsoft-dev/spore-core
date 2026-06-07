"""Consult mediation tests for :class:`SubagentTool` (issue #114, seam A1).

Mirrors the Rust ``rust/crates/spore-core/src/tools/subagent.rs`` consult tests.
``SubagentTool`` drives the full run -> Consult -> route -> budget -> handler ->
resume loop INTERNALLY (R2/R3); the parent orchestrator's model never sees the
consult. Depth-1 is preserved (R7): the handler is the orchestrator's direct
child.

Rules:

* R2 — a child :class:`RunResultConsult` is MEDIATED here, never bubbled.
* R3 — routes by ``kind``, runs the handler WITHOUT the parent model; the parent
  ultimately sees Success.
* R4 — per-kind budget: the handler runs up to ``budget`` times.
* R5a — SoftFail overflow resumes the worker with ``BudgetExhausted``.
* R5b — EscalateToHuman overflow surfaces :class:`ToolOutputWaitingForHuman`.
* R6 — no matching handler (wrong kind, or no handlers at all) -> Escalate.
* R7 — depth-1: the handler is run as the orchestrator's direct child, on the
  rendered consult request.
"""

from __future__ import annotations

from spore_core.harness import (
    AggregateUsage,
    BudgetSnapshot,
    ConsultOverflowPolicyEscalateToHuman,
    ConsultOverflowPolicySoftFail,
    ConsultHandlerEntry,
    ConsultOverflowPolicy,
    ConsultRequest,
    ConsultResponse,
    ConsultResponseAnswer,
    ConsultResponseBudgetExhausted,
    HaltReasonHumanHalted,
    HarnessRunOptions,
    HumanRequestReview,
    ReactConfig,
    PausedState,
    RunResult,
    RunResultConsult,
    RunResultFailure,
    RunResultSuccess,
    SessionId,
    SessionState,
    Task,
    TaskId,
    ToolOutputEscalate,
    ToolOutputSuccess,
    ToolOutputWaitingForHuman,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox, StandardToolRegistry, make_test_ctx
from spore_tools.tools.subagent import ContextSharingIsolated, SubagentTool

_CTX = make_test_ctx()


class ScriptedHarness:
    """Returns a queue of :class:`RunResult`s. ``resume_consult`` logs the
    :class:`ConsultResponse` it was resumed with and pops the next result (the
    default when the queue is empty is a terminal Success)."""

    def __init__(self, results: list[RunResult]) -> None:
        self._results = list(results)
        self.resume_log: list[ConsultResponse] = []

    async def run(self, options: HarnessRunOptions) -> RunResult:
        _ = options
        if not self._results:
            return RunResultFailure(
                reason=HaltReasonHumanHalted(),
                session_id=SessionId("s"),
                usage=AggregateUsage(),
                turns=0,
            )
        return self._results.pop(0)

    async def resume(self, *args: object, **kwargs: object) -> RunResult:
        raise AssertionError("resume (human) not used in these tests")

    async def resume_consult(
        self, state: PausedState, response: ConsultResponse, on_stream: object | None = None
    ) -> RunResult:
        _ = (state, on_stream)
        self.resume_log.append(response)
        if not self._results:
            return RunResultSuccess(
                output="child done after consult",
                session_id=SessionId("s"),
                usage=AggregateUsage(),
                turns=1,
            )
        return self._results.pop(0)


class RecordingHandler:
    """A handler harness that records each instruction it is run with and returns
    a fixed answer. Used to assert depth-1 routing (R3/R7)."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.seen: list[str] = []

    async def run(self, options: HarnessRunOptions) -> RunResult:
        self.seen.append(options.task.instruction)
        return RunResultSuccess(
            output=self.answer,
            session_id=SessionId("handler"),
            usage=AggregateUsage(),
            turns=1,
        )

    async def resume(self, *args: object, **kwargs: object) -> RunResult:
        raise AssertionError("handler resume not used")


def _consult_paused() -> PausedState:
    return PausedState(
        session_id=SessionId("worker"),
        task_id=TaskId("t"),
        turn_number=1,
        session_state=SessionState(),
        pending_tool_calls=[
            ToolCall(id="consult-call", name="ask_advice", input={"kind": "advice"})
        ],
        approved_results=[],
        human_request=None,
        task=Task.new("audit", SessionId("worker"), ReactConfig.per_loop(4)),
        budget_used=BudgetSnapshot(),
        child_state=None,
    )


def _consult_request(kind: str) -> ConsultRequest:
    return ConsultRequest(kind=kind, situation="drowning", attempts=2, question="what now?")


def _consult_result(kind: str) -> RunResultConsult:
    return RunResultConsult(
        request=_consult_request(kind),
        state=_consult_paused(),
        session_id=SessionId("worker"),
        usage=AggregateUsage(),
        turns=1,
    )


def _handlers(
    kind: str, handler: RecordingHandler, budget: int, overflow: ConsultOverflowPolicy
) -> dict[str, ConsultHandlerEntry]:
    return {kind: ConsultHandlerEntry(handler=handler, budget=budget, overflow=overflow)}


def _call(input_: dict) -> ToolCall:
    return ToolCall(id="parent-call-1", name="subagent", input=input_)


def _subagent_tool(harness: ScriptedHarness, handlers: dict | None = None) -> SubagentTool:
    sub = SubagentTool.new(
        name="subagent",
        description="child",
        input_schema={"type": "object"},
        timeout_seconds=5.0,
        context_sharing=ContextSharingIsolated(),
        harness=harness,
        child_registry=StandardToolRegistry(),
    )
    if handlers is not None:
        sub = sub.with_consult_handlers(handlers)
    return sub


# ---------------------------------------------------------------------------
# R2/R3 + R7: child Consult is MEDIATED (not bubbled). The handler runs (no
# parent model), the worker is resumed, and the parent ultimately sees Success.
# ---------------------------------------------------------------------------


async def test_consult_is_mediated_and_resumed_to_success() -> None:
    handler = RecordingHandler("try plan B")
    # First run() => Consult; resume_consult (default scripted) => Success.
    h = ScriptedHarness([_consult_result("advice")])
    sub = _subagent_tool(h, _handlers("advice", handler, 3, ConsultOverflowPolicySoftFail()))

    r = await sub.execute(_call({"instruction": "x"}), AllowAllSandbox(), _CTX)
    # R3: parent sees Success (the consult never reached its model).
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "child done after consult"
    # R3/R7: the handler ran exactly once, on the rendered consult request.
    assert len(handler.seen) == 1
    assert "advice" in handler.seen[0]
    assert "what now?" in handler.seen[0]
    # R3: worker resumed with the handler's answer.
    assert h.resume_log == [ConsultResponseAnswer(text="try plan B")]


# ---------------------------------------------------------------------------
# R4 + R5a: handler runs up to `budget` times; the (budget+1)th consult
# overflows. With SoftFail, the worker is resumed with BudgetExhausted.
# ---------------------------------------------------------------------------


async def test_budget_overflow_soft_fail_resumes_with_budget_exhausted() -> None:
    handler = RecordingHandler("advice answer")
    # budget = 1: run() => Consult; resume_consult => Consult again (over-budget);
    # resume_consult => Success (default).
    h = ScriptedHarness([_consult_result("advice"), _consult_result("advice")])
    sub = _subagent_tool(h, _handlers("advice", handler, 1, ConsultOverflowPolicySoftFail()))

    r = await sub.execute(_call({"instruction": "x"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)
    # R4: handler ran exactly once (budget = 1).
    assert len(handler.seen) == 1
    # R5a: first resume = Answer, second resume = BudgetExhausted.
    assert len(h.resume_log) == 2
    assert isinstance(h.resume_log[0], ConsultResponseAnswer)
    assert isinstance(h.resume_log[1], ConsultResponseBudgetExhausted)


# ---------------------------------------------------------------------------
# R5b: budget overflow with EscalateToHuman -> ToolOutputWaitingForHuman.
# ---------------------------------------------------------------------------


async def test_budget_overflow_escalate_to_human() -> None:
    handler = RecordingHandler("x")
    # budget = 0: the FIRST consult is already over budget -> escalate.
    h = ScriptedHarness([_consult_result("advice")])
    sub = _subagent_tool(h, _handlers("advice", handler, 0, ConsultOverflowPolicyEscalateToHuman()))

    r = await sub.execute(_call({"instruction": "x"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputWaitingForHuman)
    assert r.child_state.parent_tool_call_id == "parent-call-1"
    assert isinstance(r.request, HumanRequestReview)
    # Handler never ran (over budget from the start).
    assert handler.seen == []


# ---------------------------------------------------------------------------
# R6: a consult with NO matching handler (map present, wrong kind) -> Escalate.
# ---------------------------------------------------------------------------


async def test_consult_no_matching_kind_escalates() -> None:
    handler = RecordingHandler("x")
    h = ScriptedHarness([_consult_result("research")])
    sub = _subagent_tool(h, _handlers("advice", handler, 3, ConsultOverflowPolicySoftFail()))

    r = await sub.execute(_call({"instruction": "x"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputEscalate)
    assert r.signal.kind == "abort"
    assert "research" in r.signal.reason
    assert handler.seen == []


# ---------------------------------------------------------------------------
# R6 (degradation): with NO handlers installed at all, a child consult is
# treated as the no-matching-kind case -> Escalate.
# ---------------------------------------------------------------------------


async def test_consult_with_no_handlers_escalates() -> None:
    h = ScriptedHarness([_consult_result("advice")])
    sub = _subagent_tool(h)  # no handlers

    r = await sub.execute(_call({"instruction": "x"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputEscalate)


# ---------------------------------------------------------------------------
# R7: depth-1 — the handler is the orchestrator's direct child. The handler is
# run via its own `run()` with a fresh task; the child registry has no subagent
# tools (enforced at construction). This test asserts the handler ran exactly
# once per mediated consult, never nested under the worker.
# ---------------------------------------------------------------------------


async def test_handler_runs_as_orchestrator_direct_child_depth_1() -> None:
    handler = RecordingHandler("answer")
    h = ScriptedHarness([_consult_result("advice")])
    sub = _subagent_tool(h, _handlers("advice", handler, 3, ConsultOverflowPolicySoftFail()))

    await sub.execute(_call({"instruction": "x"}), AllowAllSandbox(), _CTX)
    # The handler ran exactly once, directly (its own run()), with a rendered
    # instruction — never wrapped as the worker's subagent.
    assert len(handler.seen) == 1
    assert handler.seen[0].startswith("A worker agent is requesting help")
