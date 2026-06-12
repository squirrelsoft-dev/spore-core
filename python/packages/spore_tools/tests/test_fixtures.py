"""Fixture-replay tests against ``fixtures/tools/*.json``."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.error import ToolExecutionError
from spore_tools.tools.params import (
    DeleteFileParams,
    FindFilesParams,
    GitCommitParams,
    GitDiffParams,
    GitLogParams,
    GitResetParams,
    GrepFilesParams,
    HttpGetParams,
    HttpPostParams,
    ListDirParams,
    MoveFileParams,
    ReadFileParams,
    WriteFileParams,
    parse_params,
)
from spore_core.model import ToolCall

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURES = REPO_ROOT / "fixtures" / "tools"


_PARAMS_BY_TOOL: dict[str, type] = {
    "read_file": ReadFileParams,
    "write_file": WriteFileParams,
    "list_dir": ListDirParams,
    "delete_file": DeleteFileParams,
    "move_file": MoveFileParams,
    "grep_files": GrepFilesParams,
    "find_files": FindFilesParams,
    "git_log": GitLogParams,
    "git_diff": GitDiffParams,
    "git_commit": GitCommitParams,
    "git_reset": GitResetParams,
    "http_get": HttpGetParams,
    "http_post": HttpPostParams,
}


def _param_parse_ok(tool: str, input_: dict[str, Any]) -> bool:
    if tool == "git_status":
        return True
    model = _PARAMS_BY_TOOL.get(tool)
    if model is None:
        return True
    call = ToolCall(id="fx", name=tool, input=input_)
    try:
        parsed = parse_params(model, call)
    except ToolExecutionError:
        return False
    # grep_files additionally validates that the regex compiles —
    # treat invalid regex as ``invalid_parameters`` to mirror Rust.
    if tool == "grep_files":
        try:
            re.compile(parsed.pattern)
        except re.error:
            return False
    return True


def test_fixture_replay_param_validation() -> None:
    path = FIXTURES / "param_validation.json"
    if not path.exists():
        pytest.skip("fixtures not present")
    scenarios = json.loads(path.read_text())
    assert scenarios, "expected >= 1 scenario"
    for sc in scenarios:
        actual = "ok" if _param_parse_ok(sc["tool"], sc["input"]) else "invalid_parameters"
        assert actual == sc["expected"], (
            f"scenario tool={sc['tool']} got {actual} expected {sc['expected']}"
        )


async def test_fixture_replay_output_truncation() -> None:
    path = FIXTURES / "output_truncation.json"
    if not path.exists():
        pytest.skip("fixtures not present")
    scenarios = json.loads(path.read_text())
    sb = AllowAllSandbox()
    for sc in scenarios:
        content = "x" * sc["content_length"]
        out = await sb.handle_large_output(content, "fx", sc["head_tokens"], sc["tail_tokens"])
        truncated = out.content != content
        assert truncated == sc["expects_truncated"], (
            f"truncation mismatch at content_length={sc['content_length']}"
        )


async def test_fixture_replay_subagent_scenarios() -> None:
    """Subagent fixtures — verify ``SubagentTool`` maps child run results
    onto its own :class:`ToolOutput` consistently with the Rust reference."""

    from spore_core.harness import (
        AggregateUsage,
        BudgetSnapshot,
        HaltReasonHumanHalted,
        HarnessRunOptions,
        HumanRequestClarification,
        ReactConfig,
        PausedState,
        RunResult,
        RunResultFailure,
        RunResultSuccess,
        RunResultWaitingForHuman,
        SessionState,
        Task,
        ToolOutputError,
        ToolOutputSuccess,
        ToolOutputWaitingForHuman,
        new_session_id,
        new_task_id,
    )
    from spore_core.tool_registry import StandardToolRegistry
    from spore_tools.tools.subagent import ContextSharingIsolated, SubagentTool

    class _OneShotHarness:
        def __init__(self, r: RunResult) -> None:
            self._r: RunResult | None = r

        async def run(self, options: HarnessRunOptions) -> RunResult:
            assert self._r is not None
            r, self._r = self._r, None
            return r

        async def resume(self, *_a: object, **_k: object) -> RunResult:
            raise AssertionError("unused")

    path = FIXTURES / "subagent_scenarios.json"
    if not path.exists():
        pytest.skip("fixtures not present")
    scenarios = json.loads(path.read_text())
    for sc in scenarios:
        kind = sc["child_run_result"]["kind"]
        parent_call_id = sc["parent_call_id"]
        if kind == "success":
            result: RunResult = RunResultSuccess(
                output=sc["child_run_result"]["output"],
                session_id=new_session_id("s"),
                usage=AggregateUsage(),
                turns=1,
            )
        elif kind == "failure":
            result = RunResultFailure(
                reason=HaltReasonHumanHalted(),
                session_id=new_session_id("s"),
                usage=AggregateUsage(),
                turns=1,
            )
        elif kind == "waiting_for_human":
            sid = new_session_id("s")
            paused = PausedState(
                session_id=sid,
                task_id=new_task_id("t"),
                turn_number=1,
                session_state=SessionState(),
                pending_tool_calls=[],
                approved_results=[],
                human_request=HumanRequestClarification(question="?"),
                task=Task.new(
                    instruction="x",
                    session_id=sid,
                    loop_strategy=ReactConfig.per_loop(1),
                ),
                budget_used=BudgetSnapshot(),
                child_state=None,
            )
            result = RunResultWaitingForHuman(
                state=paused, request=HumanRequestClarification(question="?")
            )
        else:  # pragma: no cover
            raise AssertionError(f"unknown kind {kind!r}")

        sub = SubagentTool.new(
            name="subagent",
            description="d",
            input_schema={"type": "object"},
            timeout_seconds=5.0,
            context_sharing=ContextSharingIsolated(),
            harness=_OneShotHarness(result),
            child_registry=StandardToolRegistry(),
        )
        out = await sub.execute(
            ToolCall(id=parent_call_id, name="subagent", input={"instruction": "x"}),
            AllowAllSandbox(),
            make_test_ctx(),
        )
        exp_kind = sc["expected"]["kind"]
        if exp_kind == "success":
            assert isinstance(out, ToolOutputSuccess)
            assert out.content == sc["expected"]["content"]
        elif exp_kind == "error":
            assert isinstance(out, ToolOutputError)
            assert out.recoverable == sc["expected"]["recoverable"]
        elif exp_kind == "waiting_for_human":
            assert isinstance(out, ToolOutputWaitingForHuman)
            assert out.child_state.parent_tool_call_id == sc["expected"]["parent_tool_call_id"]


# ===========================================================================
# Issue #81 catalogue fixtures
# ===========================================================================


def _tc(name: str, input_: dict[str, Any]) -> ToolCall:
    return ToolCall(id="fx", name=name, input=input_)


async def test_fixture_replay_edit_file_cases(tmp_path: Path) -> None:
    from spore_core.harness import ToolOutputError, ToolOutputSuccess
    from spore_tools.tools.edit import EditFileTool

    path = FIXTURES / "edit_file_cases.json"
    if not path.exists():
        pytest.skip("fixtures not present")
    cases = json.loads(path.read_text())
    assert cases
    for i, sc in enumerate(cases):
        f = tmp_path / f"case_{i}.txt"
        f.write_text(sc["initial_content"])
        out = await EditFileTool().execute(
            _tc(
                "edit_file",
                {
                    "path": str(f),
                    "old_string": sc["old_string"],
                    "new_string": sc["new_string"],
                },
            ),
            AllowAllSandbox(),
            make_test_ctx(),
        )
        exp = sc["expected"]
        if exp["kind"] == "success":
            assert isinstance(out, ToolOutputSuccess), sc["name"]
            assert f.read_text() == exp["final_content"], sc["name"]
        else:
            assert isinstance(out, ToolOutputError), sc["name"]
            assert out.recoverable == exp["recoverable"], sc["name"]
            assert exp["reason"].replace("_", " ") in out.message, sc["name"]


async def test_fixture_replay_grep_output_modes(tmp_path: Path) -> None:
    from spore_core.harness import ToolOutputSuccess
    from spore_tools.tools.search import GrepTool

    path = FIXTURES / "grep_output_modes.json"
    if not path.exists():
        pytest.skip("fixtures not present")
    cases = json.loads(path.read_text())
    assert cases
    for i, sc in enumerate(cases):
        d = tmp_path / f"grep_{i}"
        d.mkdir()
        for fname, content in sc["files"].items():
            (d / fname).write_text(content)
        out = await GrepTool().execute(
            _tc(
                "grep",
                {
                    "pattern": sc["pattern"],
                    "path": str(d),
                    "output_mode": sc["output_mode"],
                },
            ),
            AllowAllSandbox(),
            make_test_ctx(),
        )
        assert isinstance(out, ToolOutputSuccess), sc["name"]
        lines = out.content.splitlines()
        assert len(lines) == sc["expected_lines"], sc["name"]
        for needle in sc["expected_contains"]:
            assert any(needle in ln for ln in lines), f"{sc['name']}: missing {needle}"


async def test_fixture_replay_send_message_event() -> None:
    from spore_core.harness import ToolOutputError, ToolOutputSuccess
    from spore_tools.tools.message import SendMessageTool

    path = FIXTURES / "send_message_event.json"
    if not path.exists():
        pytest.skip("fixtures not present")
    cases = json.loads(path.read_text())
    assert cases
    for sc in cases:
        out = await SendMessageTool().execute(
            _tc("send_message", sc["input"]), AllowAllSandbox(), make_test_ctx()
        )
        exp = sc["expected_tool_output"]
        if exp["kind"] == "success":
            assert isinstance(out, ToolOutputSuccess), sc["name"]
            assert out.content == exp["content"], sc["name"]
            # The stream event mirrors the success content (the harness loop
            # emits it; here we assert the tool surfaces the verbatim content).
            ev = sc["expected_stream_event"]
            assert ev["kind"] == "user_message"
            assert ev["content"] == out.content, sc["name"]
        else:
            assert isinstance(out, ToolOutputError), sc["name"]
            assert out.recoverable == exp["recoverable"], sc["name"]
            assert sc["expected_stream_event"] is None


async def test_fixture_replay_todo_write(tmp_path: Path) -> None:
    from spore_core.harness import SessionId, ToolOutputSuccess
    from spore_core.storage import InMemoryStorageProvider, project_id_from_canonical_path
    from spore_core.tool_registry import ToolContext
    from spore_tools.tools.todo import TODO_STORE_KEY, TodoWriteTool

    path = FIXTURES / "todo_write.json"
    if not path.exists():
        pytest.skip("fixtures not present")
    cases = json.loads(path.read_text())
    assert cases
    for sc in cases:
        _backend = InMemoryStorageProvider()
        ctx = ToolContext(
            session_id=SessionId(f"todo-{sc['name']}"),
            project_id=project_id_from_canonical_path("/test-project"),
            run_store=_backend,
            memory_store=_backend,
        )
        tool = TodoWriteTool()
        for step in sc["steps"]:
            out = await tool.execute(_tc("todo_write", step["input"]), AllowAllSandbox(), ctx)
            assert isinstance(out, ToolOutputSuccess), sc["name"]
            assert json.loads(out.content) == step["expected_persisted"], sc["name"]
        # After all steps, the persisted blob is the LAST step's list.
        persisted = await ctx.run_store.get(ctx.session_id, TODO_STORE_KEY)
        assert persisted == sc["steps"][-1]["expected_persisted"], sc["name"]


async def test_fixture_replay_escalation_tools() -> None:
    from spore_core.harness import (
        HarnessSignalAbort,
        HarnessSignalEnterPlanMode,
        HarnessSignalExitPlanMode,
        ToolOutputAwaitingClarification,
        ToolOutputEscalate,
    )
    from spore_tools.tools.control import (
        AbortTool,
        AskUserQuestionTool,
        EnterPlanModeTool,
        ExitPlanModeTool,
    )

    path = FIXTURES / "escalation_tools.json"
    if not path.exists():
        pytest.skip("fixtures not present")
    cases = json.loads(path.read_text())
    assert cases
    tools = {
        "enter_plan_mode": EnterPlanModeTool(),
        "exit_plan_mode": ExitPlanModeTool(),
        "ask_user_question": AskUserQuestionTool(),
        "abort": AbortTool(),
    }
    for sc in cases:
        tool = tools[sc["tool"]]
        out = await tool.execute(_tc(sc["tool"], sc["input"]), AllowAllSandbox(), make_test_ctx())
        exp = sc["expected"]
        if exp["tool_output_kind"] == "escalate":
            assert isinstance(out, ToolOutputEscalate), sc["name"]
            sig = exp["signal"]
            if sig["kind"] == "enter_plan_mode":
                assert isinstance(out.signal, HarnessSignalEnterPlanMode)
                assert out.signal.context == sig["context"]
            elif sig["kind"] == "exit_plan_mode":
                assert isinstance(out.signal, HarnessSignalExitPlanMode)
                assert out.signal.plan.tasks == sig["plan"]["tasks"]
                assert out.signal.plan.rationale == sig["plan"].get("rationale", "")
            elif sig["kind"] == "abort":
                assert isinstance(out.signal, HarnessSignalAbort)
                assert out.signal.reason == sig["reason"]
            else:  # pragma: no cover
                raise AssertionError(sig["kind"])
        elif exp["tool_output_kind"] == "awaiting_clarification":
            assert isinstance(out, ToolOutputAwaitingClarification), sc["name"]
            assert out.question == exp["question"]
            assert out.options == exp["options"]
        else:  # pragma: no cover
            raise AssertionError(exp["tool_output_kind"])
