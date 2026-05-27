"""Regression: the task instruction must reach the agent as the first user
message (issue #57).

Mirrors ``task_instruction_delivered_as_first_user_message`` in
``rust/crates/spore-core/tests/e2e_scenarios.rs``. Unlike ``MockAgent``, which
ignores its ``Context``, the agent here records every assembled ``Context`` so
we can assert the model actually receives the prompt. The harness is backed by
the real ``StandardCompactionAdapter`` context manager (exactly like a live
run): the adapter mirrors ``session.messages`` and ignores ``task`` on
``assemble``, so without the harness seeding the instruction the captured
first-turn context is EMPTY and this test fails — which is the bug we fix.
"""

from __future__ import annotations

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    CompactionConfig,
    HarnessConfig,
    HarnessRunOptions,
    LoopStrategyReAct,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    StandardCompactionAdapter,
    StandardContextManager,
    StandardHarness,
    Task,
)
from spore_core.agent import Context, FinalResponse, TurnResult
from spore_core.model import ProviderInfo, Role, TokenUsage


class _StubModel:
    """Minimal ``ModelInterface`` — the adapter only reaches ``count_tokens`` /
    ``provider`` on the assemble path, and those return constants."""

    async def call(self, request: object) -> object:  # pragma: no cover - unused
        raise NotImplementedError

    async def call_streaming(self, request: object) -> object:  # pragma: no cover - unused
        raise NotImplementedError

    async def count_tokens(self, request: object) -> int:
        return 0

    def provider(self) -> ProviderInfo:
        return ProviderInfo(name="stub", model_id="stub", context_window=200_000)


class _CapturingAgent:
    """Records every ``Context`` it is handed, then returns a scripted final
    response."""

    def __init__(self) -> None:
        self.seen: list[Context] = []

    async def turn(self, context: Context) -> TurnResult:
        self.seen.append(context)
        return FinalResponse(
            content="DONE",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    def id(self) -> AgentId:
        return AgentId("capture")


def _rich_adapter() -> StandardCompactionAdapter:
    cfg = CompactionConfig(
        threshold=0.80,
        preserve_recent_n=2,
        head_tail_tokens=64,
        max_compaction_attempts=2,
    )
    return StandardCompactionAdapter(StandardContextManager(_StubModel(), compaction=cfg))


async def test_task_instruction_delivered_as_first_user_message() -> None:
    agent = _CapturingAgent()
    harness = StandardHarness(
        HarnessConfig(
            agent=agent,
            tool_registry=ScriptedToolRegistry(),
            sandbox=AllowAllSandbox(),
            context_manager=_rich_adapter(),
            termination_policy=AlwaysContinuePolicy(),
        )
    )

    instruction = "summarize the quarterly payment report"
    task = Task.new(
        instruction,
        SessionId("seed-test"),
        LoopStrategyReAct(max_iterations=4),
    )
    result = await harness.run(HarnessRunOptions(task))
    assert isinstance(result, RunResultSuccess), f"expected Success, got {result!r}"

    assert agent.seen, "agent must have been invoked at least once"
    first = agent.seen[0]
    has_user_instruction = any(
        m.role == Role.USER and getattr(m.content, "text", None) == instruction
        for m in first.messages
    )
    assert has_user_instruction, (
        "first-turn context must contain a User message equal to the task "
        f"instruction; got messages: {first.messages!r}"
    )
