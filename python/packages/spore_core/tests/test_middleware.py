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
    ReactConfig,
    RunResultSuccess,
    SessionId,
    SessionState,
    Task,
    ToolOutputSuccess,
)
from spore_core.middleware import (
    AlreadyRegisteredError,
    HookContext,
    HookContextBeforeSession,
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
from spore_core.observability import (
    InMemoryObservabilityProvider,
    ObservabilityProvider,
    PatchTypeDanglingToolCall,
    SpanKind,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _task() -> Task:
    return Task.new(
        "test task",
        SessionId("sess"),
        ReactConfig.per_loop(5),
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


# ── Patch observability (issue #28) ────────────────────────────────────────


def _wired_patch(obs: InMemoryObservabilityProvider) -> PatchToolCallsMiddleware:
    return PatchToolCallsMiddleware("noop").with_observability(obs)


async def _run_patch(mw: PatchToolCallsMiddleware, calls: list[ToolCall]) -> MiddlewareDecision:
    """Drive identity capture (BeforeSession) then a BeforeTool fire directly
    on the middleware so the test owns the calls list."""
    await mw.handle(HookContextBeforeSession(task=_task(), session_id=_sid()))
    return await mw.handle(HookContextBeforeTool(calls=calls, turn_number=1))


# R1 + R3: every patch emits exactly one warn-level span recording both the
# original and the patched parameters.
async def test_patch_emits_one_warn_span_with_original_and_patched() -> None:
    obs = InMemoryObservabilityProvider()
    mw = _wired_patch(obs)
    calls = [_tc(name="", input={"command": "ls"}, id="c1")]
    d = await _run_patch(mw, calls)
    assert isinstance(d, MiddlewareContinueWithModification)
    spans = obs.patch_spans(_sid())
    assert len(spans) == 1
    p = spans[0]
    assert p.level.value == "warn"
    # R3: original preserved, patched reflects the dispatched call.
    assert p.original_parameters == {"command": "ls"}
    assert p.patched_parameters == {"command": "ls"}
    # The mutation that triggered the patch (the name) is visible on the span.
    assert p.tool_name == "noop"
    assert p.call_id == "c1"


# R2: no patch needed → no span, decision is plain Continue.
async def test_no_patch_emits_no_span() -> None:
    obs = InMemoryObservabilityProvider()
    mw = _wired_patch(obs)
    calls = [_tc(name="shell", id="c1")]
    d = await _run_patch(mw, calls)
    assert isinstance(d, MiddlewareContinue)
    assert obs.patch_spans(_sid()) == []
    trace = await obs.get_trace(_sid())
    assert all(s.base.kind is not SpanKind.PATCH for s in trace)


# R4: the empty-name repair is classified as DanglingToolCall.
async def test_empty_name_classified_as_dangling_tool_call() -> None:
    obs = InMemoryObservabilityProvider()
    mw = _wired_patch(obs)
    calls = [_tc(name="", id="c1")]
    await _run_patch(mw, calls)
    p = obs.patch_spans(_sid())[0]
    assert isinstance(p.patch_type, PatchTypeDanglingToolCall)
    assert p.patch_type.reason == "empty tool name"


# R5: the patch event is present in get_trace.
async def test_trace_contains_patch_event() -> None:
    obs = InMemoryObservabilityProvider()
    mw = _wired_patch(obs)
    calls = [_tc(name="", id="c1")]
    await _run_patch(mw, calls)
    trace = await obs.get_trace(_sid())
    assert any(s.base.kind is SpanKind.PATCH for s in trace)


# R9: a batch of N patched calls emits N patch spans (one per patch).
async def test_batch_emits_one_span_per_patch() -> None:
    obs = InMemoryObservabilityProvider()
    mw = _wired_patch(obs)
    calls = [
        _tc(name="", id="c1"),
        _tc(name="shell", id="c2"),
        _tc(name="  ", id="c3"),
    ]
    await _run_patch(mw, calls)
    spans = obs.patch_spans(_sid())
    assert len(spans) == 2  # c1 and c3, not c2
    assert {s.call_id for s in spans} == {"c1", "c3"}


# Silent without a provider: patching still works and emits nothing.
async def test_patch_without_observability_is_silent() -> None:
    mw = PatchToolCallsMiddleware("noop")
    calls = [_tc(name="", id="c1")]
    d = await _run_patch(mw, calls)
    assert isinstance(d, MiddlewareContinueWithModification)
    assert calls[0].name == "noop"


# R10 regression: still registers at the highest BeforeTool priority and still
# hooks BeforeTool. Also satisfies the ObservabilityProvider collaborator type.
def test_patch_middleware_priority_is_highest_before_tool() -> None:
    import sys

    mw = PatchToolCallsMiddleware("noop")
    assert mw.priority() == -sys.maxsize + 1
    assert HookPoint.BEFORE_TOOL in mw.hooks()
    assert isinstance(InMemoryObservabilityProvider(), ObservabilityProvider)


# R6/R7/R8: patch metrics roll up: count, rate (2/4 = 0.5), and per-tool.
async def test_patch_metrics_count_rate_and_by_tool() -> None:
    from spore_core.guide_registry import SessionOutcomeSuccess
    from spore_core.memory import Timestamp
    from spore_core.observability import SpanBase, ToolCallSpan, new_span_id

    obs = InMemoryObservabilityProvider()
    mw = _wired_patch(obs)

    def _tool_call_span(span_id: str, tool: str) -> ToolCallSpan:
        base = SpanBase(
            span_id=new_span_id(span_id),
            session_id=_sid(),
            task_id=_task().id,
            kind=SpanKind.TOOL_CALL,
            started_at=Timestamp("2026-05-16T00:00:00Z"),
            ended_at=Timestamp("2026-05-16T00:00:00Z"),
        )
        return ToolCallSpan(
            base=base,
            tool_name=tool,
            call_id=span_id,
            parameters_size_bytes=0,
            output_size_bytes=0,
            truncated=False,
            sandbox_mode="none",
        )

    # 4 tool calls total, 2 of which (both "shell") will be patched.
    for sid, tool in [("tc1", "shell"), ("tc2", "shell"), ("tc3", "edit"), ("tc4", "edit")]:
        obs.emit_tool_call(_tool_call_span(sid, tool))
    # Two empty-name shell calls get patched to the fallback "noop".
    calls = [_tc(name="", id="tc1"), _tc(name="", id="tc2")]
    await _run_patch(mw, calls)

    obs.set_session_outcome(_sid(), SessionOutcomeSuccess())
    m = await obs.get_session_metrics(_sid())
    assert m is not None
    assert m.patch_count == 2
    assert abs(m.patch_rate - 0.5) < 1e-6  # 2 / 4
    assert m.patches_by_tool.get("noop") == 2
    assert "edit" not in m.patches_by_tool


# R7: zero tool calls → patch_rate is 0.0, never a divide-by-zero.
async def test_patch_rate_zero_when_no_tool_calls() -> None:
    from spore_core.guide_registry import SessionOutcomeSuccess

    obs = InMemoryObservabilityProvider()
    mw = _wired_patch(obs)
    calls = [_tc(name="", id="c1")]
    await _run_patch(mw, calls)
    obs.set_session_outcome(_sid(), SessionOutcomeSuccess())
    m = await obs.get_session_metrics(_sid())
    assert m is not None
    assert m.patch_count == 1
    assert m.patch_rate == 0.0


# R11: fixture replay against the shared cross-language fixture.
async def test_fixture_replay_patch_events() -> None:
    from spore_core.guide_registry import SessionOutcomeSuccess

    path = Path(__file__).resolve().parents[4] / "fixtures" / "patch" / "patch_events_basic.json"
    case = json.loads(path.read_text())

    obs = InMemoryObservabilityProvider()
    mw = PatchToolCallsMiddleware(case["fallback_name"]).with_observability(obs)

    calls = [ToolCall(id=c["id"], name=c["name"], input=c["input"]) for c in case["input_calls"]]
    await _run_patch(mw, calls)

    spans = obs.patch_spans(_sid())
    assert len(spans) == len(case["expected_patches"])
    for exp in case["expected_patches"]:
        found = next(p for p in spans if p.call_id == exp["call_id"])
        assert found.tool_name == exp["tool_name"]
        assert found.original_parameters == exp["original"]
        assert found.patched_parameters == exp["patched"]
        assert exp["patch_type"] == "dangling_tool_call"
        assert isinstance(found.patch_type, PatchTypeDanglingToolCall)

    obs.set_session_outcome(_sid(), SessionOutcomeSuccess())
    m = await obs.get_session_metrics(_sid())
    assert m is not None
    assert m.patch_count == case["expected_patch_count"]
    for tool, n in case["expected_patches_by_tool"].items():
        assert m.patches_by_tool.get(tool) == n
