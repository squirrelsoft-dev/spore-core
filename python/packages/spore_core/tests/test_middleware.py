"""Tests for :class:`StandardMiddlewareChain` — issue #11.

Mirrors the Rust unit tests in
``rust/crates/spore-core/src/middleware.rs`` ``tests`` module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spore_core.harness import (
    AggregateUsage,
    BudgetLimits,
    HarnessToolResult,
    HumanRequestClarification,
    HumanRequestToolApproval,
    LoopStrategyReAct,
    RunResultSuccess,
    SessionId,
    SessionState,
    Task,
    ToolOutputSuccess,
)
from spore_core.middleware import (
    AlreadyRegisteredError,
    HookContext,
    HookContextBeforeTool,
    HookPoint,
    IllegalDecisionError,
    LoopDetectionMiddleware,
    MiddlewareChain,
    MiddlewareContinue,
    MiddlewareContinueWithModification,
    MiddlewareDecision,
    MiddlewareForceAnotherTurn,
    MiddlewareHalt,
    MiddlewareSurfaceToHuman,
    NoHooksError,
    PatchToolCallsMiddleware,
    PreCompletionChecklistMiddleware,
    StandardMiddlewareChain,
    TokenBudgetMiddleware,
    TracingMiddleware,
)
from spore_core.model import ToolCall


# ── Helpers ────────────────────────────────────────────────────────────────


def _task() -> Task:
    return Task.new(
        "test task",
        SessionId("sess"),
        LoopStrategyReAct(max_iterations=5),
        budget=BudgetLimits(),
    )


def _sid() -> SessionId:
    return SessionId("sess")


def _tc(name: str = "edit", input: dict | None = None, id: str = "c1") -> ToolCall:
    return ToolCall(id=id, name=name, input=input if input is not None else {})


def _tr(call_id: str = "c1", content: str = "ok") -> HarnessToolResult:
    return HarnessToolResult(call_id=call_id, output=ToolOutputSuccess(content=content))


# ── Scripted recording middleware ──────────────────────────────────────────


class Scripted:
    """Programmable middleware that records every fire and returns a
    canned decision."""

    def __init__(
        self,
        name: str,
        hooks: list[HookPoint],
        priority: int = 0,
        decision: MiddlewareDecision | None = None,
    ) -> None:
        self._name = name
        self._hooks = list(hooks)
        self._priority = priority
        self._decision: MiddlewareDecision = decision or MiddlewareContinue()
        self.fired: list[HookPoint] = []

    async def handle(self, ctx: HookContext) -> MiddlewareDecision:
        self.fired.append(ctx.point())
        return self._decision

    def hooks(self) -> list[HookPoint]:
        return list(self._hooks)

    def priority(self) -> int:
        return self._priority

    def name(self) -> str:
        return self._name


# ── Rule: register validates hooks list and uniqueness ─────────────────────


async def test_register_rejects_empty_hooks() -> None:
    chain = StandardMiddlewareChain()
    with pytest.raises(NoHooksError):
        await chain.register(Scripted("m", []))


async def test_register_rejects_duplicate_name() -> None:
    chain = StandardMiddlewareChain()
    await chain.register(Scripted("m", [HookPoint.BEFORE_TURN]))
    with pytest.raises(AlreadyRegisteredError):
        await chain.register(Scripted("m", [HookPoint.BEFORE_TURN]))


# ── Rule: before hooks run in ascending priority ───────────────────────────


async def test_before_hooks_run_in_ascending_priority() -> None:
    chain = StandardMiddlewareChain()
    fired_order: list[str] = []

    class P:
        def __init__(self, name: str, priority: int) -> None:
            self._name = name
            self._priority = priority

        async def handle(self, ctx: HookContext) -> MiddlewareDecision:
            fired_order.append(self._name)
            return MiddlewareContinue()

        def hooks(self) -> list[HookPoint]:
            return [HookPoint.BEFORE_TURN]

        def priority(self) -> int:
            return self._priority

        def name(self) -> str:
            return self._name

    await chain.register(P("c", 100))
    await chain.register(P("b", 10))
    await chain.register(P("a", -50))

    d = await chain.fire_before_turn(SessionState(), 1)
    assert isinstance(d, MiddlewareContinue)
    assert fired_order == ["a", "b", "c"]


# ── Rule: after hooks run in descending priority ───────────────────────────


async def test_after_hooks_run_in_descending_priority() -> None:
    chain = StandardMiddlewareChain()
    fired_order: list[str] = []

    class P:
        def __init__(self, name: str, priority: int) -> None:
            self._name = name
            self._priority = priority

        async def handle(self, ctx: HookContext) -> MiddlewareDecision:
            fired_order.append(self._name)
            return MiddlewareContinue()

        def hooks(self) -> list[HookPoint]:
            return [HookPoint.AFTER_TOOL]

        def priority(self) -> int:
            return self._priority

        def name(self) -> str:
            return self._name

    await chain.register(P("low", 1))
    await chain.register(P("high", 100))
    await chain.register(P("mid", 50))

    await chain.fire_after_tool([_tc()], [_tr()])
    assert fired_order == ["high", "mid", "low"]


# ── Rule: first Halt stops the chain ───────────────────────────────────────


async def test_halt_stops_chain() -> None:
    chain = StandardMiddlewareChain()
    halter = Scripted(
        "halt",
        [HookPoint.BEFORE_TURN],
        priority=1,
        decision=MiddlewareHalt(reason="stop"),
    )
    after = Scripted("after", [HookPoint.BEFORE_TURN], priority=100)
    await chain.register(halter)
    await chain.register(after)
    d = await chain.fire_before_turn(SessionState(), 1)
    assert isinstance(d, MiddlewareHalt)
    assert d.reason == "stop"
    assert after.fired == []


# ── Rule: SurfaceToHuman first-wins on BeforeTool ──────────────────────────


async def test_surface_to_human_first_wins_on_before_tool() -> None:
    chain = StandardMiddlewareChain()
    req = HumanRequestToolApproval(calls=[_tc(name="shell")], risk_level="high")
    first = Scripted(
        "first",
        [HookPoint.BEFORE_TOOL],
        priority=1,
        decision=MiddlewareSurfaceToHuman(request=req),
    )
    second = Scripted("second", [HookPoint.BEFORE_TOOL], priority=2)
    await chain.register(first)
    await chain.register(second)
    calls = [_tc(name="shell")]
    d = await chain.fire_before_tool(calls, 1)
    assert isinstance(d, MiddlewareSurfaceToHuman)
    assert second.fired == []


# ── Rule: SurfaceToHuman illegal outside BeforeTool / BeforeCompletion ─────


async def test_surface_to_human_illegal_on_before_turn() -> None:
    chain = StandardMiddlewareChain()
    bad = Scripted(
        "bad",
        [HookPoint.BEFORE_TURN],
        priority=1,
        decision=MiddlewareSurfaceToHuman(
            request=HumanRequestClarification(question="?"),
        ),
    )
    await chain.register(bad)
    d = await chain.fire_before_turn(SessionState(), 1)
    assert isinstance(d, MiddlewareHalt)
    assert "SurfaceToHuman" in d.reason


# ── Rule: ForceAnotherTurn concatenated, chain continues ───────────────────


async def test_force_another_turn_concatenates_and_continues() -> None:
    chain = StandardMiddlewareChain()
    await chain.register(
        Scripted(
            "a",
            [HookPoint.BEFORE_COMPLETION],
            priority=1,
            decision=MiddlewareForceAnotherTurn(inject="first"),
        )
    )
    await chain.register(
        Scripted(
            "b",
            [HookPoint.BEFORE_COMPLETION],
            priority=2,
            decision=MiddlewareForceAnotherTurn(inject="second"),
        )
    )
    d = await chain.fire_before_completion("done", 3, SessionState())
    assert isinstance(d, MiddlewareForceAnotherTurn)
    assert "first" in d.inject
    assert "second" in d.inject
    # Newline-joined per spec.
    assert d.inject == "first\nsecond"


# ── Rule: ForceAnotherTurn illegal outside BeforeCompletion ────────────────


async def test_force_another_turn_illegal_on_before_turn() -> None:
    chain = StandardMiddlewareChain()
    await chain.register(
        Scripted(
            "bad",
            [HookPoint.BEFORE_TURN],
            priority=1,
            decision=MiddlewareForceAnotherTurn(inject="x"),
        )
    )
    d = await chain.fire_before_turn(SessionState(), 1)
    assert isinstance(d, MiddlewareHalt)
    assert "ForceAnotherTurn" in d.reason


# ── Rule: PatchToolCalls renames empty names ───────────────────────────────


async def test_patch_tool_calls_renames_empty_name() -> None:
    chain = StandardMiddlewareChain()
    await chain.register(PatchToolCallsMiddleware("noop"))
    calls = [_tc(name="")]
    d = await chain.fire_before_tool(calls, 1)
    assert isinstance(d, MiddlewareContinueWithModification)
    assert calls[0].name == "noop"


# ── Rule: PatchToolCalls runs before other BeforeTool middleware ───────────


async def test_patch_tool_calls_runs_before_other_before_tool_middleware() -> None:
    chain = StandardMiddlewareChain()
    observed_name: list[str] = []

    class Observer:
        async def handle(self, ctx: HookContext) -> MiddlewareDecision:
            assert isinstance(ctx, HookContextBeforeTool)
            for c in ctx.calls:
                observed_name.append(c.name)
            return MiddlewareContinue()

        def hooks(self) -> list[HookPoint]:
            return [HookPoint.BEFORE_TOOL]

        def priority(self) -> int:
            return 0

        def name(self) -> str:
            return "observer"

    await chain.register(PatchToolCallsMiddleware("noop"))
    await chain.register(Observer())
    calls = [_tc(name="")]
    d = await chain.fire_before_tool(calls, 1)
    assert isinstance(d, MiddlewareContinueWithModification)
    # Observer must have seen the patched call, proving PatchToolCalls
    # ran first.
    assert observed_name == ["noop"]


# ── Rule: AfterTool mutation propagates via ContinueWithModification ───────


async def test_loop_detection_annotates_after_threshold() -> None:
    chain = StandardMiddlewareChain()
    await chain.register(LoopDetectionMiddleware("edit", 2))
    calls = [_tc(name="edit", input={"path": "/tmp/foo.txt"})]

    # First fire: under threshold.
    results = [_tr()]
    d = await chain.fire_after_tool(calls, results)
    assert isinstance(d, MiddlewareContinue)

    # Second fire reaches threshold and annotates.
    results = [_tr()]
    d = await chain.fire_after_tool(calls, results)
    assert isinstance(d, MiddlewareContinueWithModification)
    assert isinstance(results[0].output, ToolOutputSuccess)
    assert "[loop-detection]" in results[0].output.content


# ── Rule: PreCompletionChecklist forces another turn when missing ──────────


async def test_pre_completion_checklist_forces_another_turn() -> None:
    chain = StandardMiddlewareChain()
    await chain.register(PreCompletionChecklistMiddleware(["tests passed"]))
    d = await chain.fire_before_completion("done", 1, SessionState())
    assert isinstance(d, MiddlewareForceAnotherTurn)
    assert "tests passed" in d.inject

    d = await chain.fire_before_completion("all tests passed", 1, SessionState())
    assert isinstance(d, MiddlewareContinue)


# ── Rule: TokenBudget halts when exhausted ─────────────────────────────────


async def test_token_budget_halts_when_exhausted() -> None:
    chain = StandardMiddlewareChain()
    budget = TokenBudgetMiddleware(100)
    await chain.register(budget)
    assert isinstance(await chain.fire_before_turn(SessionState(), 1), MiddlewareContinue)
    budget.record(150)
    d = await chain.fire_before_turn(SessionState(), 2)
    assert isinstance(d, MiddlewareHalt)
    assert "token budget" in d.reason


# ── Edge: BeforeSession and AfterSession fire end-to-end ───────────────────


async def test_session_boundary_hooks_fire() -> None:
    chain = StandardMiddlewareChain()
    tracing = TracingMiddleware()
    await chain.register(tracing)
    task = _task()
    sid = _sid()
    await chain.fire_before_session(task, sid)
    result = RunResultSuccess(
        output="done",
        session_id=sid,
        usage=AggregateUsage(),
        turns=1,
    )
    await chain.fire_after_session(result, sid)
    log = tracing.entries()
    assert any(p == HookPoint.BEFORE_SESSION for p, _ in log)
    assert any(p == HookPoint.AFTER_SESSION for p, _ in log)


# ── TracingMiddleware lowest priority — fires first on before hooks ────────


async def test_tracing_middleware_fires_first_on_before_hooks() -> None:
    chain = StandardMiddlewareChain()
    tracing = TracingMiddleware()
    other = Scripted("other", [HookPoint.BEFORE_TURN], priority=0)
    await chain.register(other)
    await chain.register(tracing)
    await chain.fire_before_turn(SessionState(), 7)
    log = tracing.entries()
    assert log == [(HookPoint.BEFORE_TURN, 7)]
    assert other.fired == [HookPoint.BEFORE_TURN]


# ── HookPoint helpers ──────────────────────────────────────────────────────


def test_hook_point_is_before_after() -> None:
    assert HookPoint.BEFORE_SESSION.is_before()
    assert HookPoint.BEFORE_TURN.is_before()
    assert HookPoint.BEFORE_TOOL.is_before()
    assert HookPoint.BEFORE_COMPLETION.is_before()
    assert HookPoint.AFTER_TOOL.is_after()
    assert HookPoint.AFTER_SESSION.is_after()
    assert HookPoint.BEFORE_TOOL.allows_surface_to_human()
    assert HookPoint.BEFORE_COMPLETION.allows_surface_to_human()
    assert not HookPoint.BEFORE_TURN.allows_surface_to_human()
    assert HookPoint.BEFORE_COMPLETION.allows_force_another_turn()
    assert not HookPoint.BEFORE_TOOL.allows_force_another_turn()


# ── IllegalDecisionError shape ─────────────────────────────────────────────


def test_illegal_decision_error_attributes() -> None:
    e = IllegalDecisionError("m", HookPoint.BEFORE_TURN, "ForceAnotherTurn")
    assert e.name == "m"
    assert e.hook is HookPoint.BEFORE_TURN
    assert e.decision == "ForceAnotherTurn"
    assert "before_turn" in str(e)


# ── Chain is a MiddlewareChain ─────────────────────────────────────────────


def test_chain_satisfies_protocol() -> None:
    chain = StandardMiddlewareChain()
    assert isinstance(chain, MiddlewareChain)


# ── Fixture replay ─────────────────────────────────────────────────────────


async def test_fixture_replay_before_completion_checklist() -> None:
    path = Path(__file__).resolve().parents[4] / "fixtures" / "middleware" / "checklist_basic.json"
    raw = json.loads(path.read_text())
    chain = StandardMiddlewareChain()
    await chain.register(PreCompletionChecklistMiddleware(raw["required"]))
    d = await chain.fire_before_completion(raw["response"], 1, SessionState())
    if raw["expected"] == "continue":
        assert isinstance(d, MiddlewareContinue)
    elif raw["expected"] == "force_another_turn":
        assert isinstance(d, MiddlewareForceAnotherTurn)
    else:
        pytest.fail(f"unexpected fixture outcome: {raw['expected']}")
