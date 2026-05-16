"""Fixture-replay tests against ``fixtures/tools/*.json``."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from spore_core.tool_registry import AllowAllSandbox
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
        truncated = out.summary != content
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
        LoopStrategyReAct,
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
                    loop_strategy=LoopStrategyReAct(max_iterations=1),
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
