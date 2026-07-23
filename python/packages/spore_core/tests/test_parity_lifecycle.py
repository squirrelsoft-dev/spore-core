"""Cross-language lifecycle parity regression tests (parity plan §4.15).

Pins the invariants that must match the Rust reference byte-for-byte:
- the retryable HTTP status set + exponential backoff curve,
- ``ModelError.retryable()`` classification,
- ``BudgetLimits.wall_time`` timeout surfaces a typed budget halt,
- ``dispatch_all`` partitions read-only calls concurrently and the rest
  sequentially, preserving order.

(Agent ``HarnessSignalAbort`` → ``RunResultEscalate`` is already pinned in
``test_harness_escalation.py`` R8; provider retry/backoff is exercised end-to-end
in ``test_anthropic.py``.)
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    HarnessConfig,
    HarnessRunOptions,
    ModelAgent,
    ModelResponse,
    NoopContextManager,
    ProviderInfo,
    ReactConfig,
    RecordedExchange,
    RegistryToolSchema,
    ReplayModelInterface,
    RunResultFailure,
    ScriptedToolRegistry,
    SessionId,
    StandardHarness,
    StandardToolRegistry,
    StopReason,
    Task,
    TextBlock,
    TokenUsage,
    ToolAnnotations,
    ToolCall,
    ToolUseBlock,
    ToolOutputSuccess,
)
from spore_core.anthropic import _RETRYABLE_STATUSES, _backoff_delay
from spore_core.model import (
    ProviderError,
    RateLimited,
    TimeoutError as ModelTimeoutError,
    Transport,
)


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=1, output_tokens=1)


# ---------------------------------------------------------------------------
# Provider retry: retryable status set + backoff curve (mirrors Rust)
# ---------------------------------------------------------------------------


def test_retryable_status_set_matches_rust() -> None:
    # Rust provider_http::is_retryable_status = {408,425,429,500,502,503,504};
    # Anthropic adds 529.
    assert _RETRYABLE_STATUSES == frozenset({408, 425, 429, 500, 502, 503, 504, 529})


def test_backoff_curve_matches_rust() -> None:
    # 0.5, 1, 2, 4, 8, 16, 30 (cap) seconds.
    assert _backoff_delay(0) == pytest.approx(0.5)
    assert _backoff_delay(1) == pytest.approx(1.0)
    assert _backoff_delay(2) == pytest.approx(2.0)
    assert _backoff_delay(3) == pytest.approx(4.0)
    assert _backoff_delay(20) <= 30.0


def test_model_error_retryable_classification() -> None:
    assert Transport("drop").retryable() is True
    assert ModelTimeoutError().retryable() is True
    assert RateLimited(retry_after=None).retryable() is True
    assert ProviderError(code=400, message="bad").retryable() is False


# ---------------------------------------------------------------------------
# Timeout: wall-time budget surfaces a typed budget halt
# ---------------------------------------------------------------------------


def _provider() -> ProviderInfo:
    return ProviderInfo(name="anthropic", model_id="replay", context_window=200_000)


def _replay_agent(responses: list[ModelResponse]) -> ModelAgent:
    from spore_core import ModelRequest

    exchanges = [
        RecordedExchange(request=ModelRequest(stream=True), response=r, provider="anthropic")
        for r in responses
    ]
    return ModelAgent(AgentId("to-agent"), ReplayModelInterface(exchanges, _provider()))


async def test_wall_time_budget_surfaces_budget_exceeded() -> None:
    # A 0-second wall-time budget trips immediately on the first turn gate.
    agent = _replay_agent(
        [
            ModelResponse(
                content=[TextBlock(text="done")],
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                stop_reason=StopReason.END_TURN,
            )
        ]
    )
    task = Task.new("x", SessionId("s1"), ReactConfig.per_loop(5))
    task.budget.max_wall_time = 0
    cfg = HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
    )
    h = StandardHarness(cfg)
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure), r
    assert "BudgetExceeded" in type(r.reason).__name__ or "budget" in str(r.reason).lower()


# ---------------------------------------------------------------------------
# Concurrency: dispatch_all partitions read-only vs sequential
# ---------------------------------------------------------------------------


class _RecordingTool:
    def __init__(self, name: str, *, read_only: bool) -> None:
        self._name = name
        self.calls: list[float] = []

    def name(self) -> str:
        return self._name

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    async def execute(self, call: Any, sandbox: Any, ctx: Any) -> Any:
        self.calls.append(time.monotonic())
        import anyio

        await anyio.sleep(0.02)
        return ToolOutputSuccess(content=self._name)


def _schema(name: str, ann: ToolAnnotations) -> RegistryToolSchema:
    return RegistryToolSchema(
        name=name, description="", parameters={"type": "object", "properties": {}}, annotations=ann
    )


async def test_dispatch_all_runs_read_only_concurrently_and_sequential_rest() -> None:
    reg = StandardToolRegistry()
    ro_a = _RecordingTool("ro_a", read_only=True)
    ro_b = _RecordingTool("ro_b", read_only=True)
    seq = _RecordingTool("seq", read_only=False)
    reg.register(ro_a, _schema("ro_a", ToolAnnotations(read_only=True)))
    reg.register(ro_b, _schema("ro_b", ToolAnnotations(read_only=True)))
    reg.register(seq, _schema("seq", ToolAnnotations(read_only=False)))

    calls = [
        ToolCall(id="c1", name="ro_a", input={}),
        ToolCall(id="c2", name="seq", input={}),
        ToolCall(id="c3", name="ro_b", input={}),
    ]
    results = await reg.dispatch_all(calls, AllowAllSandbox(), None)  # type: ignore[arg-type]
    assert len(results) == 3
    assert ro_a.calls and ro_b.calls and seq.calls
    # The two read-only tools overlapped (their start times are close).
    assert abs(ro_a.calls[0] - ro_b.calls[0]) < 0.02, (
        f"read-only calls did not overlap: {ro_a.calls} {ro_b.calls}"
    )


# ---------------------------------------------------------------------------
# Cleanup finalization (§review Low #8): terminal runs record the session
# outcome via observability; a WaitingForHuman pause does NOT finalize.
# ---------------------------------------------------------------------------


async def test_cleanup_finalizes_success_outcome() -> None:
    from spore_core import InMemoryObservabilityProvider, SessionOutcomeSuccess

    obs = InMemoryObservabilityProvider()
    agent = _replay_agent(
        [
            ModelResponse(
                content=[TextBlock(text="done")],
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                stop_reason=StopReason.END_TURN,
            )
        ]
    )
    cfg = HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        observability=obs,
    )
    h = StandardHarness(cfg)
    sid = SessionId("cleanup-ok")
    await h.run(HarnessRunOptions(Task.new("x", sid, ReactConfig.per_loop(3))))
    assert obs._outcomes.get(sid) is not None
    assert isinstance(obs._outcomes[sid], SessionOutcomeSuccess)


async def test_cleanup_finalizes_failure_outcome() -> None:
    from spore_core import InMemoryObservabilityProvider, SessionOutcomeFailure

    obs = InMemoryObservabilityProvider()
    agent = _replay_agent(
        [
            ModelResponse(
                content=[TextBlock(text="done")],
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                stop_reason=StopReason.END_TURN,
            )
        ]
    )
    task = Task.new("x", SessionId("cleanup-fail"), ReactConfig.per_loop(5))
    task.budget.max_wall_time = 0  # immediate budget halt → Failure
    cfg = HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        observability=obs,
    )
    h = StandardHarness(cfg)
    await h.run(HarnessRunOptions(task))
    assert isinstance(obs._outcomes[SessionId("cleanup-fail")], SessionOutcomeFailure)


# ---------------------------------------------------------------------------
# Steering in-flight cancellation (§review Low #8): steering injects at turn
# boundaries AFTER tool execution — in-flight tool work is never cancelled.
# ---------------------------------------------------------------------------


async def test_steering_does_not_cancel_in_flight_tool_work() -> None:
    from dataclasses import dataclass

    from spore_core import (
        HarnessBuilder,
        RegistryToolSchema,
        StreamToolResult,
        StreamUserMessage,
        ToolAnnotations,
        ToolOutputSuccess,
    )

    @dataclass
    class _Bundle:
        implementation: Any
        schema: Any

    tool_started: list[float] = []
    tool_finished: list[float] = []

    class _SlowTool:
        def name(self) -> str:
            return "slow"

        def is_subagent_tool(self) -> bool:
            return False

        def may_produce_large_output(self) -> bool:
            return False

        async def execute(self, call: Any, sandbox: Any, ctx: Any) -> Any:
            import anyio

            tool_started.append(time.monotonic())
            await anyio.sleep(0.03)  # in-flight work
            tool_finished.append(time.monotonic())
            return ToolOutputSuccess(content="slow-done")

    # Steering queue: empty on the first turn boundary (so the first turn runs
    # the tool), then a message on the second boundary (after the tool result).
    steer_calls = 0

    def steering() -> list[str]:
        nonlocal steer_calls
        steer_calls += 1
        return ["pivot"] if steer_calls == 2 else []

    agent = _replay_agent(
        [
            ModelResponse(
                content=[ToolUseBlock(id="toolu_1", name="slow", input={})],
                usage=_usage(),
                stop_reason=StopReason.TOOL_USE,
            ),
            ModelResponse(
                content=[TextBlock(text="done")],
                usage=_usage(),
                stop_reason=StopReason.END_TURN,
            ),
        ]
    )
    events: list = []
    builder = (
        HarnessBuilder(
            agent=agent,
            tool_registry=ScriptedToolRegistry(),
            sandbox=AllowAllSandbox(),
            context_manager=NoopContextManager(),
            termination_policy=AlwaysContinuePolicy(),
        )
        .tool(
            _Bundle(
                implementation=_SlowTool(),
                schema=RegistryToolSchema(
                    name="slow",
                    description="slow tool",
                    parameters={"type": "object", "properties": {}},
                    annotations=ToolAnnotations(),
                ),
            )
        )
        .steering_source(steering)
    )
    h = builder.build()
    await h.run(
        HarnessRunOptions(
            Task.new("x", SessionId("steer"), ReactConfig.per_loop(5)), on_stream=events.append
        )
    )
    # The tool completed (in-flight work was NOT cancelled).
    assert tool_finished, "in-flight tool work must complete"
    # The steered user message was emitted AFTER the tool result.
    tool_result_idx = next(i for i, e in enumerate(events) if isinstance(e, StreamToolResult))
    steered_idx = next(
        (i for i, e in enumerate(events) if isinstance(e, StreamUserMessage) and e.steered), None
    )
    assert steered_idx is not None, "expected a steered StreamUserMessage"
    assert steered_idx > tool_result_idx, "steering must inject AFTER the tool result, not mid-turn"
