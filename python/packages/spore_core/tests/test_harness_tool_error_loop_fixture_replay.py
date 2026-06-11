"""Loop-replay integration test for the consecutive-recoverable-tool-error
breaker (issue #137).

Loads ``fixtures/model_responses/harness/tool_error_loop.jsonl`` — a recorded
trace in which the model repeatedly emits the SAME malformed ``add_task`` tool
call (the gemma ``task_list``/``add_task``-without-``description`` scenario) — and
drives a :class:`StandardHarness` with a ReAct leaf. The scripted tool registry
returns an identical ``ToolOutputError(recoverable=True)`` for every dispatch.
With ``error_loop_threshold = 3`` (N) and a generous turn budget (50), the ONLY
thing that can stop the run is the breaker, which hard-stops at 2N (6 identical
errors) and resolves the leaf's ``Fail`` behavior into
``RunResultFailure(reason=HaltReasonToolErrorLoop)`` WITHOUT burning the rest of
the budget.

Must produce the SAME outcome as the Rust / TypeScript / Go implementations —
never edit the fixture to make a failing implementation pass (see
``fixtures/README.md``).
"""

from __future__ import annotations

from pathlib import Path

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetExhaustedFail,
    EscalationModeAutonomous,
    HaltReasonToolErrorLoop,
    HarnessConfig,
    HarnessRunOptions,
    NoopContextManager,
    ProviderInfo,
    ReactConfig,
    ReplayModelInterface,
    RunResultFailure,
    SessionId,
    StandardHarness,
    Task,
    ToolCall,
    ToolOutput,
    ToolOutputError,
    ToolSchema,
)
from spore_core.agent import ModelAgent


def _fixture_path() -> Path:
    here = Path(__file__).resolve()
    return here.parents[4] / "fixtures" / "model_responses" / "harness" / "tool_error_loop.jsonl"


class _AlwaysErrorRegistry:
    """Every dispatch of the malformed ``add_task`` call returns the same
    recoverable error, regardless of args (mirrors the Rust replay test's
    ``always_recoverable_error``)."""

    def __init__(self, message: str) -> None:
        self._message = message
        self.call_count = 0

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        self.call_count += 1
        return ToolOutputError(message=self._message, recoverable=True)

    def is_always_halt(self, tool_name: str) -> bool:
        return False

    def schemas(self) -> list[ToolSchema]:
        return []


async def test_tool_error_loop_breaker_hard_stops_at_two_n() -> None:
    jsonl = _fixture_path().read_text()
    replay = ReplayModelInterface.from_jsonl(
        jsonl,
        ProviderInfo(name="ollama", model_id="fixture", context_window=200_000),
    )
    agent = ModelAgent(AgentId("fixture-agent"), replay)

    tool_registry = _AlwaysErrorRegistry("missing required parameter `description`")

    config = HarnessConfig(
        agent=agent,
        tool_registry=tool_registry,  # type: ignore[arg-type]
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        # N == 3 -> inject at 3 identical errors, hard-stop at 6.
        error_loop_threshold=3,
        # Autonomous so the leaf's Fail behavior produces a terminal Failure (not
        # a HITL pause) at the 2N hard stop.
        escalation_mode=EscalationModeAutonomous(),
    )
    harness = StandardHarness(config)

    # A bare ReAct leaf with a generous budget (50) and Fail behavior — so the
    # ONLY thing that can stop the run early is the error-loop breaker.
    task = Task.new(
        "add a task to the task list",
        SessionId("tool-error-loop-session"),
        ReactConfig(
            budget=ReactConfig.per_loop(50).budget,
            behavior=BudgetExhaustedFail(),
            agent="",
            toolset="",
        ),
    )

    r = await harness.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonToolErrorLoop)
    assert r.reason.tool == "add_task"
    assert r.reason.consecutive_errors == 6, "hard stop at 2N == 6"
    assert r.session_id == SessionId("tool-error-loop-session")
    # The breaker stopped EARLY: exactly 2N turns, far below the budget.
    assert r.turns == 6, "exactly 2N turns consumed before the hard stop"
    assert r.turns < 50, f"budget NOT fully burned, got {r.turns}"
    # The breaker stops AT the 2N dispatch — it does not append/continue past it
    # — so the registry saw exactly 2N == 6 calls (the 7th fixture line is unused
    # headroom: the breaker, not fixture exhaustion, ended the run).
    assert tool_registry.call_count == 6, (
        "tool dispatched exactly 2N times, then the breaker stopped"
    )
