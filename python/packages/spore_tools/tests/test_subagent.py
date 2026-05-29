"""Tests for :class:`SubagentTool`."""

from __future__ import annotations

import pytest

from spore_core.harness import (
    AggregateUsage,
    BudgetSnapshot,
    HaltReasonHumanHalted,
    HarnessRunOptions,
    HumanRequestClarification,
    LoopStrategyReAct,
    PausedState,
    RunResult,
    RunResultFailure,
    RunResultSuccess,
    RunResultWaitingForHuman,
    SessionState,
    Task,
    TaskId,
    ToolOutputError,
    ToolOutputSuccess,
    ToolOutputWaitingForHuman,
    new_session_id,
    new_task_id,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import (
    AllowAllSandbox,
    StandardToolRegistry,
    SubagentMock,
    ToolAnnotations,
    ToolSchema,
    make_test_ctx,
)
from spore_tools.tools.subagent import (
    BuildError,
    ContextSharingIsolated,
    SubagentTool,
)

_CTX = make_test_ctx()


class _ScriptedHarness:
    """Pop-front harness: returns the next scripted :class:`RunResult`."""

    def __init__(self, results: list[RunResult]) -> None:
        self._results = list(results)

    async def run(self, options: HarnessRunOptions) -> RunResult:
        if not self._results:
            raise AssertionError("scripted harness exhausted")
        return self._results.pop(0)

    async def resume(self, *args: object, **kwargs: object) -> RunResult:
        raise AssertionError("resume not used in these tests")


def _call(input_: dict) -> ToolCall:
    return ToolCall(id="parent-call-1", name="subagent", input=input_)


def _subagent_tool(harness: _ScriptedHarness, *, timeout: float = 5.0) -> SubagentTool:
    return SubagentTool.new(
        name="subagent",
        description="child",
        input_schema={"type": "object"},
        timeout_seconds=timeout,
        context_sharing=ContextSharingIsolated(),
        harness=harness,
        child_registry=StandardToolRegistry(),
    )


async def test_subagent_success_maps_to_tool_success() -> None:
    h = _ScriptedHarness(
        [
            RunResultSuccess(
                output="child done",
                session_id=new_session_id("s"),
                usage=AggregateUsage(),
                turns=1,
            )
        ]
    )
    sub = _subagent_tool(h)
    r = await sub.execute(_call({"instruction": "do it"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "child done"


async def test_subagent_failure_maps_to_recoverable_error() -> None:
    h = _ScriptedHarness(
        [
            RunResultFailure(
                reason=HaltReasonHumanHalted(),
                session_id=new_session_id("s"),
                usage=AggregateUsage(),
                turns=1,
            )
        ]
    )
    sub = _subagent_tool(h)
    r = await sub.execute(_call({"instruction": "x"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_subagent_waiting_for_human_propagates_with_parent_call_id() -> None:
    sid = new_session_id("s")
    tid = TaskId(new_task_id("t"))
    paused = PausedState(
        session_id=sid,
        task_id=tid,
        turn_number=1,
        session_state=SessionState(),
        pending_tool_calls=[],
        approved_results=[],
        human_request=HumanRequestClarification(question="yes?"),
        task=Task.new(
            instruction="x",
            session_id=sid,
            loop_strategy=LoopStrategyReAct(max_iterations=1),
        ),
        budget_used=BudgetSnapshot(),
        child_state=None,
    )
    h = _ScriptedHarness(
        [
            RunResultWaitingForHuman(
                state=paused,
                request=HumanRequestClarification(question="yes?"),
            )
        ]
    )
    sub = _subagent_tool(h)
    r = await sub.execute(_call({"instruction": "x"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputWaitingForHuman)
    assert r.child_state.parent_tool_call_id == "parent-call-1"


async def test_construction_rejects_child_with_subagent_tools() -> None:
    child_reg = StandardToolRegistry()
    child_reg.register(
        SubagentMock("nested"),
        ToolSchema(
            name="nested",
            description="n",
            parameters={"type": "object"},
            annotations=ToolAnnotations(),
        ),
    )
    h = _ScriptedHarness([])
    with pytest.raises(BuildError):
        SubagentTool.new(
            name="subagent",
            description="child",
            input_schema={"type": "object"},
            timeout_seconds=1.0,
            context_sharing=ContextSharingIsolated(),
            harness=h,
            child_registry=child_reg,
        )


async def test_missing_instruction_returns_recoverable_error() -> None:
    h = _ScriptedHarness([])
    sub = _subagent_tool(h, timeout=1.0)
    r = await sub.execute(_call({}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
