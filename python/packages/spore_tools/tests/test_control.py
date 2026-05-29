"""Tests for the Tier-3 control tools (#81): enter/exit plan mode, ask, abort."""

from __future__ import annotations

from spore_core.harness import (
    HarnessSignalAbort,
    HarnessSignalEnterPlanMode,
    HarnessSignalExitPlanMode,
    ToolOutputAwaitingClarification,
    ToolOutputError,
    ToolOutputEscalate,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.control import (
    AbortTool,
    AskUserQuestionTool,
    EnterPlanModeTool,
    ExitPlanModeTool,
)

_CTX = make_test_ctx()


def _call(name: str, input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=name, input=input_)


async def test_enter_plan_mode_escalates() -> None:
    r = await EnterPlanModeTool().execute(
        _call("enter_plan_mode", {"context": "seed"}), AllowAllSandbox(), _CTX
    )
    assert isinstance(r, ToolOutputEscalate)
    assert isinstance(r.signal, HarnessSignalEnterPlanMode)
    assert r.signal.context == "seed"


async def test_enter_plan_mode_context_defaults() -> None:
    r = await EnterPlanModeTool().execute(_call("enter_plan_mode", {}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputEscalate)
    assert isinstance(r.signal, HarnessSignalEnterPlanMode)
    assert r.signal.context == ""


async def test_exit_plan_mode_escalates_with_plan() -> None:
    r = await ExitPlanModeTool().execute(
        _call("exit_plan_mode", {"plan": {"tasks": ["a", "b"], "rationale": "because"}}),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputEscalate)
    assert isinstance(r.signal, HarnessSignalExitPlanMode)
    assert r.signal.plan.tasks == ["a", "b"]
    assert r.signal.plan.rationale == "because"


async def test_exit_plan_mode_rationale_defaults() -> None:
    r = await ExitPlanModeTool().execute(
        _call("exit_plan_mode", {"plan": {"tasks": ["x"]}}), AllowAllSandbox(), _CTX
    )
    assert isinstance(r, ToolOutputEscalate)
    assert isinstance(r.signal, HarnessSignalExitPlanMode)
    assert r.signal.plan.tasks == ["x"]
    assert r.signal.plan.rationale == ""


async def test_ask_user_question_with_options() -> None:
    r = await AskUserQuestionTool().execute(
        _call("ask_user_question", {"question": "which?", "options": ["a", "b"]}),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputAwaitingClarification)
    assert r.question == "which?"
    assert r.options == ["a", "b"]


async def test_ask_user_question_free_form() -> None:
    r = await AskUserQuestionTool().execute(
        _call("ask_user_question", {"question": "free form?"}), AllowAllSandbox(), _CTX
    )
    assert isinstance(r, ToolOutputAwaitingClarification)
    assert r.question == "free form?"
    assert r.options is None


async def test_abort_escalates() -> None:
    r = await AbortTool().execute(_call("abort", {"reason": "stop now"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputEscalate)
    assert isinstance(r.signal, HarnessSignalAbort)
    assert r.signal.reason == "stop now"


async def test_abort_missing_reason_is_recoverable_error() -> None:
    r = await AbortTool().execute(_call("abort", {}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
