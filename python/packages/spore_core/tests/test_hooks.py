"""Tests for the lifecycle hook system — issue #69.

Mirrors the Rust unit tests in ``rust/crates/spore-core/src/hooks.rs`` and
replays the cross-language fixtures in ``fixtures/hooks/`` (ground truth).
"""

from __future__ import annotations

import asyncio
import json
import stat
import tempfile
from pathlib import Path

import pytest

from spore_core.agent import Context as ContextBlock
from spore_core.context import CompactionPreserveHints
from spore_core.hooks import (
    ALL_EVENTS,
    CommandHook,
    FireOutcome,
    FunctionHook,
    HookBlock,
    HookContinue,
    HookDeny,
    HookError,
    HookEvent,
    HookInject,
    HookMutate,
    OnLoopStartContext,
    OnPlanCreatedContext,
    OnSubagentSpawnContext,
    PlanArtifact,
    PostTurnContext,
    PreCompactContext,
    PreToolUseContext,
    PreTurnContext,
    StandardHookChain,
    StopContext,
    TurnOutput,
    context_to_payload,
    hook_decision_to_dict,
    parse_hook_decision,
)
from spore_core.agent import AgentId, FinalResponse, MockAgent
from spore_core.harness import (
    AllowAllSandbox,
    AlwaysContinuePolicy,
    HarnessBuilder,
    HarnessRunOptions,
    LoopStrategyReAct,
    NoopContextManager,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    Task,
)
from spore_core.model import TokenUsage

_FIXTURES = Path(__file__).resolve().parents[4] / "fixtures" / "hooks"


def _sid() -> SessionId:
    return SessionId("s1")


def _fn(name: str, events: list[HookEvent], decision) -> FunctionHook:
    return FunctionHook(name, events, lambda _ctx, d=decision: d)


def _stop_ctx() -> StopContext:
    return StopContext(
        session_id=_sid(),
        turn_number=1,
        last_output=TurnOutput(),
        task_instruction="x",
        session_state=None,
    )


# ── classification predicates ──────────────────────────────────────────────


def test_seventeen_events() -> None:
    assert len(ALL_EVENTS) == 17
    assert len(set(ALL_EVENTS)) == 17


def test_pre_events_are_mutable() -> None:
    pre = {
        HookEvent.PRE_TURN,
        HookEvent.PRE_TOOL_USE,
        HookEvent.ON_LOOP_START,
        HookEvent.ON_RESUME,
        HookEvent.ON_TASK_ADVANCE,
        HookEvent.ON_SUBAGENT_SPAWN,
        HookEvent.PRE_COMPACT,
    }
    for ev in ALL_EVENTS:
        assert ev.is_pre() is (ev in pre)
        assert ev.is_mutable() is (ev in pre)


def test_sync_async_only_classification() -> None:
    assert HookEvent.STOP.is_sync_only()
    assert not HookEvent.STOP.is_async_only()
    assert HookEvent.ON_PAUSE.is_async_only()
    assert HookEvent.POST_COMPACT.is_async_only()
    assert HookEvent.POST_TURN.is_sync_only() is False
    assert HookEvent.POST_TURN.is_async_only() is False


def test_can_block_and_deny() -> None:
    assert HookEvent.STOP.can_block()
    assert HookEvent.POST_TURN.can_block() is False
    assert HookEvent.PRE_TOOL_USE.can_deny()
    assert HookEvent.ON_SUBAGENT_SPAWN.can_deny()
    assert HookEvent.PRE_TURN.can_deny() is False


# ── R3 / R25: registration-order firing, per-event filtering ────────────────


async def test_fires_in_registration_order() -> None:
    order: list[str] = []
    chain = StandardHookChain()
    for label in ("a", "b", "c"):

        def make(lbl: str):
            def handler(_ctx):
                order.append(lbl)
                return HookContinue()

            return handler

        chain.register(FunctionHook(label, [HookEvent.POST_TURN], make(label)))
    await chain.fire(PostTurnContext(_sid(), 1, TurnOutput()))
    assert order == ["a", "b", "c"]


async def test_hook_only_fires_for_invoked_event() -> None:
    fired: list[str] = []
    chain = StandardHookChain()
    chain.register(
        FunctionHook(
            "multi",
            [HookEvent.POST_TURN, HookEvent.PRE_TURN],
            lambda ctx: fired.append(type(ctx).__name__) or HookContinue(),
        )
    )
    await chain.fire(PostTurnContext(_sid(), 1, TurnOutput()))
    assert fired == ["PostTurnContext"]


# ── R1 / R16: pre-hook mutation in place ────────────────────────────────────


async def test_pre_tool_use_mutates_input_in_place() -> None:
    chain = StandardHookChain()

    def handler(ctx):
        ctx.tool_input = {"path": "/safe"}
        return HookContinue()

    chain.register(FunctionHook("mut", [HookEvent.PRE_TOOL_USE], handler))
    ctx = PreToolUseContext(_sid(), 1, "read_file", {"path": "/etc/passwd"})
    out = await chain.fire(ctx)
    assert out == FireOutcome.cont()
    assert ctx.tool_input == {"path": "/safe"}


# ── R2: pre-hook chain threads mutation to the next hook ────────────────────


async def test_pre_hook_chain_threads_mutation() -> None:
    chain = StandardHookChain()

    def first(ctx):
        ctx.tool_input = {"v": 1}
        return HookContinue()

    def second(ctx):
        ctx.tool_input = {"v": ctx.tool_input.get("v", 0) + 1}
        return HookContinue()

    chain.register(FunctionHook("first", [HookEvent.PRE_TOOL_USE], first))
    chain.register(FunctionHook("second", [HookEvent.PRE_TOOL_USE], second))
    ctx = PreToolUseContext(_sid(), 1, "t", {})
    await chain.fire(ctx)
    assert ctx.tool_input == {"v": 2}


# ── R6: Mutate decision replaces the mutable field ──────────────────────────


async def test_mutate_decision_replaces_field() -> None:
    chain = StandardHookChain()
    chain.register(_fn("m", [HookEvent.PRE_TOOL_USE], HookMutate(data={"replaced": True})))
    ctx = PreToolUseContext(_sid(), 1, "t", {"orig": 1})
    await chain.fire(ctx)
    assert ctx.tool_input == {"replaced": True}


async def test_mutate_on_loop_start_coerces_string() -> None:
    chain = StandardHookChain()
    chain.register(_fn("m", [HookEvent.ON_LOOP_START], HookMutate(data="new instruction")))
    ctx = OnLoopStartContext(_sid(), "old", None)  # type: ignore[arg-type]
    await chain.fire(ctx)
    assert ctx.task_instruction == "new instruction"


async def test_mutate_coercion_failure_is_handler_failed() -> None:
    """A Mutate whose data cannot be coerced into the target field type raises
    HookError with kind ``handler_failed`` (matches Rust/TS/Go)."""
    chain = StandardHookChain()
    # PreTurn's mutable field is a ContextBlock; a scalar cannot validate into it.
    chain.register(_fn("m", [HookEvent.PRE_TURN], HookMutate(data=123)))
    ctx = PreTurnContext(_sid(), 1, ContextBlock())
    with pytest.raises(HookError) as exc:
        await chain.fire(ctx)
    assert exc.value.kind == "handler_failed"


# ── R15: PreToolUse deny ────────────────────────────────────────────────────


async def test_pre_tool_use_deny() -> None:
    chain = StandardHookChain()
    chain.register(_fn("deny", [HookEvent.PRE_TOOL_USE], HookDeny(reason="blocked path")))
    ctx = PreToolUseContext(_sid(), 1, "t", {})
    out = await chain.fire(ctx)
    assert out == FireOutcome.deny("blocked path")


# ── R10 / R12: sync post-hook (Stop) block ──────────────────────────────────


async def test_stop_hook_block() -> None:
    chain = StandardHookChain()
    chain.register(_fn("verify", [HookEvent.STOP], HookBlock(reason="tests failing")))
    out = await chain.fire(_stop_ctx())
    assert out == FireOutcome.block("tests failing")


# ── R13: Stop all-continue terminates ───────────────────────────────────────


async def test_stop_hook_all_continue() -> None:
    chain = StandardHookChain()
    chain.register(_fn("ok", [HookEvent.STOP], HookContinue()))
    assert await chain.fire(_stop_ctx()) == FireOutcome.cont()


async def test_stop_no_hooks_continue() -> None:
    chain = StandardHookChain()
    assert await chain.fire(_stop_ctx()) == FireOutcome.cont()


# ── R8: Stop registered async is rejected ───────────────────────────────────


def test_stop_async_rejected() -> None:
    chain = StandardHookChain()
    hook = FunctionHook("s", [HookEvent.STOP], lambda _c: HookContinue()).async_mode()
    with pytest.raises(HookError) as exc:
        chain.register(hook)
    assert exc.value.kind == "sync_only_event"


# ── R9: OnPause registered sync is rejected ─────────────────────────────────


def test_on_pause_sync_rejected() -> None:
    chain = StandardHookChain()
    hook = FunctionHook("p", [HookEvent.ON_PAUSE], lambda _c: HookContinue())
    with pytest.raises(HookError) as exc:
        chain.register(hook)
    assert exc.value.kind == "async_only_event"


# ── R4 / R17 / R24: illegal Block on a non-blocking event rejected at fire ──


async def test_illegal_block_on_post_turn() -> None:
    chain = StandardHookChain()
    chain.register(_fn("bad", [HookEvent.POST_TURN], HookBlock(reason="no")))
    with pytest.raises(HookError) as exc:
        await chain.fire(PostTurnContext(_sid(), 1, TurnOutput()))
    assert exc.value.kind == "illegal_decision"


# ── R5: Deny outside PreToolUse/OnSubagentSpawn rejected ─────────────────────


async def test_deny_validation() -> None:
    chain_ok = StandardHookChain()
    chain_ok.register(_fn("d", [HookEvent.PRE_TOOL_USE], HookDeny(reason="x")))
    assert (await chain_ok.fire(PreToolUseContext(_sid(), 1, "t", {}))).kind == "deny"

    chain_bad = StandardHookChain()
    chain_bad.register(_fn("d", [HookEvent.PRE_TURN], HookDeny(reason="x")))
    with pytest.raises(HookError) as exc:
        await chain_bad.fire(PreTurnContext(_sid(), 1, ContextBlock()))
    assert exc.value.kind == "illegal_decision"


# ── R11: async fire-and-forget not awaited ──────────────────────────────────


async def test_async_post_hook_fire_and_forget() -> None:
    ran = asyncio.Event()
    chain = StandardHookChain()

    async def slow(_ctx):
        ran.set()
        return HookContinue()

    chain.register(FunctionHook("log", [HookEvent.POST_TURN], slow).async_mode())
    # Returns Continue immediately; the async hook never affects the outcome.
    out = await chain.fire(PostTurnContext(_sid(), 1, TurnOutput()))
    assert out == FireOutcome.cont()


# ── R7: Inject aggregation ──────────────────────────────────────────────────


async def test_inject_aggregates_newline_joined() -> None:
    chain = StandardHookChain()
    chain.register(_fn("i1", [HookEvent.PRE_TURN], HookInject(context="one")))
    chain.register(_fn("i2", [HookEvent.PRE_TURN], HookInject(context="two")))
    out = await chain.fire(PreTurnContext(_sid(), 1, ContextBlock()))
    assert out == FireOutcome.inject("one\ntwo")


# ── R23: FunctionHook runs the callable ─────────────────────────────────────


async def test_function_hook_runs() -> None:
    def handler(ctx):
        ctx.task_instruction += " [checked]"
        return HookContinue()

    chain = StandardHookChain()
    chain.register(FunctionHook("f", [HookEvent.ON_LOOP_START], handler))
    ctx = OnLoopStartContext(_sid(), "do work", None)  # type: ignore[arg-type]
    await chain.fire(ctx)
    assert ctx.task_instruction == "do work [checked]"


# ── R18-R21: CommandHook roundtrip + nonzero exit ───────────────────────────


async def test_command_hook_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "hook.sh"
        script.write_text(
            '#!/bin/sh\ncat >/dev/null\necho \'{"decision":"block","reason":"cmd says no"}\'\n'
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        hook = CommandHook("cmd", [HookEvent.STOP], "sh", [str(script)])
        chain = StandardHookChain()
        chain.register(hook)
        out = await chain.fire(_stop_ctx())
        assert out == FireOutcome.block("cmd says no")


async def test_command_hook_nonzero_exit_errors() -> None:
    hook = CommandHook("cmd", [HookEvent.STOP], "sh", ["-c", "exit 7"])
    chain = StandardHookChain()
    chain.register(hook)
    with pytest.raises(HookError) as exc:
        await chain.fire(_stop_ctx())
    assert exc.value.kind == "command_failed"


async def test_command_hook_malformed_stdout_errors() -> None:
    hook = CommandHook("cmd", [HookEvent.STOP], "sh", ["-c", "echo 'not json'"])
    chain = StandardHookChain()
    chain.register(hook)
    with pytest.raises(HookError) as exc:
        await chain.fire(_stop_ctx())
    assert exc.value.kind == "command_output_invalid"


# ── HookDecision wire format ────────────────────────────────────────────────


def test_hook_decision_wire_format() -> None:
    assert hook_decision_to_dict(HookContinue()) == {"decision": "continue"}
    assert hook_decision_to_dict(HookBlock(reason="r")) == {
        "decision": "block",
        "reason": "r",
    }
    assert hook_decision_to_dict(HookInject(context="c")) == {
        "decision": "inject",
        "context": "c",
    }
    assert hook_decision_to_dict(HookDeny(reason="d")) == {
        "decision": "deny",
        "reason": "d",
    }
    assert hook_decision_to_dict(HookMutate(data={"k": 1})) == {
        "decision": "mutate",
        "data": {"k": 1},
    }


def test_hook_decision_roundtrip_from_json() -> None:
    for payload in (
        {"decision": "continue"},
        {"decision": "block", "reason": "x"},
        {"decision": "inject", "context": "y"},
        {"decision": "deny", "reason": "z"},
        {"decision": "mutate", "data": {"a": 1}},
    ):
        parsed = parse_hook_decision(payload)
        assert hook_decision_to_dict(parsed) == payload


# ── Deferred-event fire methods work in isolation ───────────────────────────


async def test_deferred_on_plan_created_mutates() -> None:
    chain = StandardHookChain()

    def handler(ctx):
        ctx.plan.tasks.append("extra")
        return HookContinue()

    chain.register(FunctionHook("plan", [HookEvent.ON_PLAN_CREATED], handler))
    ctx = OnPlanCreatedContext(_sid(), PlanArtifact(tasks=["a"]))
    await chain.fire(ctx)
    assert ctx.plan.tasks == ["a", "extra"]


async def test_deferred_subagent_spawn_deny() -> None:
    chain = StandardHookChain()
    chain.register(_fn("ss", [HookEvent.ON_SUBAGENT_SPAWN], HookDeny(reason="no spawn")))
    ctx = OnSubagentSpawnContext(_sid(), "child task", "react")
    assert await chain.fire(ctx) == FireOutcome.deny("no spawn")


async def test_pre_compact_mutates_hints() -> None:
    chain = StandardHookChain()

    def handler(ctx):
        ctx.preserve_hints.keep_recent_file_list = False
        return HookContinue()

    chain.register(FunctionHook("pc", [HookEvent.PRE_COMPACT], handler))
    ctx = PreCompactContext(_sid(), CompactionPreserveHints())
    await chain.fire(ctx)
    assert ctx.preserve_hints.keep_recent_file_list is False


# ── command-handler payload shape ───────────────────────────────────────────


def test_stop_context_payload_shape() -> None:
    payload = context_to_payload(_stop_ctx())
    assert payload == {
        "session_id": "s1",
        "turn_number": 1,
        "last_output": {"text": "", "had_tool_calls": False},
        "task_instruction": "x",
        "session_state": None,
    }


# ============================================================================
# Fixture replay — ground truth (fixtures/hooks/)
# ============================================================================


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


def test_fixture_hook_decision_wire() -> None:
    data = _load("hook_decision_wire.json")
    for case in data["cases"]:
        wire = case["json"]
        parsed = parse_hook_decision(wire)
        assert hook_decision_to_dict(parsed) == wire, case["name"]


async def test_fixture_pre_tool_use_mutation() -> None:
    data = _load("pre_tool_use_mutation.json")
    for case in data["cases"]:
        chain = StandardHookChain()
        for i, dec in enumerate(case["hook_decisions"]):
            decision = parse_hook_decision(dec)
            chain.register(
                FunctionHook(f"h{i}", [HookEvent.PRE_TOOL_USE], lambda _c, d=decision: d)
            )
        ctx = PreToolUseContext(
            _sid(), 1, case["tool_name"], json.loads(json.dumps(case["tool_input"]))
        )
        outcome = await chain.fire(ctx)
        exp = case["expected"]
        if exp["outcome"] == "continue":
            assert outcome.kind == "continue", case["name"]
            assert ctx.tool_input == exp["tool_input"], case["name"]
        else:
            assert outcome.kind == "deny", case["name"]
            assert outcome.reason == exp["reason"], case["name"]


async def test_fixture_stop_block_basic() -> None:
    """Replay the per-run Stop-block / max_stop_blocks cap behavior.

    Models the harness loop: each turn fires one Stop decision in sequence; a
    ``block`` under the cap consumes one block and continues; once the cap is
    reached the next block is ignored and the loop terminates.
    """
    data = _load("stop_block_basic.json")
    for case in data["cases"]:
        max_stop_blocks = case["max_stop_blocks"]
        decisions = case["hook_decisions"]
        blocks = 0
        terminated_by = "continue"
        for dec in decisions:
            decision = parse_hook_decision(dec)
            chain = StandardHookChain()
            chain.register(FunctionHook("stop", [HookEvent.STOP], lambda _c, d=decision: d))
            outcome = await chain.fire(_stop_ctx())
            if outcome.kind == "block":
                if blocks >= max_stop_blocks:
                    terminated_by = "cap"
                    break
                blocks += 1
                continue
            terminated_by = "continue"
            break
        assert blocks == case["expected"]["blocks"], case["name"]
        assert terminated_by == case["expected"]["terminated_by"], case["name"]


async def test_fixture_command_handler_io() -> None:
    """Replay the CommandHook stdin/stdout contract through real subprocesses."""
    data = _load("command_handler_io.json")
    event_ctx = {
        "stop": lambda: StopContext(
            session_id=SessionId("sess-1"),
            turn_number=3,
            last_output=TurnOutput(text="I'm done", had_tool_calls=False),
            task_instruction="make the tests pass",
            session_state=None,
        ),
        "pre_tool_use": lambda: PreToolUseContext(
            SessionId("sess-1"), 1, "read_file", {"path": "/etc/passwd"}
        ),
    }
    for case in data["cases"]:
        ctx = event_ctx[case["event"]]()

        # Verify the stdin payload we would send matches the pinned shape.
        if "expected_stdin" in case:
            payload = {
                "event": case["event"],
                "context": context_to_payload(ctx),
            }
            assert payload == case["expected_stdin"], case["name"]

        exit_code = case.get("exit_code", 0)
        stdout = case["stdout"]
        # A handler that consumes stdin, then emits the pinned stdout / exit.
        script = f"cat >/dev/null\nprintf '%s' {json.dumps(stdout)}\nexit {exit_code}\n"
        hook = CommandHook("cmd", [ctx.EVENT], "sh", ["-c", script])
        chain = StandardHookChain()
        chain.register(hook)

        if "expected_error" in case:
            with pytest.raises(HookError) as exc:
                await chain.fire(ctx)
            assert exc.value.kind == case["expected_error"], case["name"]
        else:
            outcome_decision = parse_hook_decision(case["expected_decision"])
            # Fire through the chain and confirm the parsed decision drives the
            # expected outcome.
            out = await chain.fire(ctx)
            if isinstance(outcome_decision, HookBlock):
                assert out == FireOutcome.block(outcome_decision.reason), case["name"]
            elif isinstance(outcome_decision, HookDeny):
                assert out == FireOutcome.deny(outcome_decision.reason), case["name"]


# ============================================================================
# Harness wiring — Stop hook is the live loop-wired event (issue #69)
# ============================================================================


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=1, output_tokens=1)


def _harness(chain: StandardHookChain, *, max_stop_blocks: int = 8, turns: int = 5):
    agent = MockAgent(AgentId("test"))
    for _ in range(turns):
        agent.push(FinalResponse(content="done", usage=_usage()))
    builder = (
        HarnessBuilder(
            agent,
            ScriptedToolRegistry(),
            AllowAllSandbox(),
            NoopContextManager(),
            AlwaysContinuePolicy(),
        )
        .hooks(chain)
        .max_stop_blocks(max_stop_blocks)
    )
    return builder.build()


def _task(max_iter: int = 10) -> Task:
    return Task.new("do something", SessionId("s1"), LoopStrategyReAct(max_iterations=max_iter))


async def test_harness_stop_block_then_continue() -> None:
    """A Stop hook that blocks once then allows termination runs an extra turn."""
    calls = {"n": 0}

    def handler(_ctx):
        calls["n"] += 1
        return HookBlock(reason="not done") if calls["n"] == 1 else HookContinue()

    chain = StandardHookChain()
    chain.register(FunctionHook("v", [HookEvent.STOP], handler))
    h = _harness(chain)
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.turns == 2  # blocked once, then terminated on the next turn


async def test_harness_stop_all_continue_terminates_immediately() -> None:
    chain = StandardHookChain()
    chain.register(FunctionHook("v", [HookEvent.STOP], lambda _c: HookContinue()))
    h = _harness(chain)
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.turns == 1


async def test_harness_max_stop_blocks_cap() -> None:
    """A Stop hook that always blocks terminates anyway after max_stop_blocks."""
    chain = StandardHookChain()
    chain.register(FunctionHook("v", [HookEvent.STOP], lambda _c: HookBlock(reason="again")))
    h = _harness(chain, max_stop_blocks=3, turns=10)
    r = await h.run(HarnessRunOptions(_task(max_iter=20)))
    assert isinstance(r, RunResultSuccess)
    # 3 honored blocks → 3 extra turns; the 4th Stop is over the cap and
    # terminates: turn 1 + 3 continued turns = 4 turns total.
    assert r.turns == 4


async def test_harness_no_hooks_terminates_normally() -> None:
    chain = StandardHookChain()
    h = _harness(chain)
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.turns == 1
