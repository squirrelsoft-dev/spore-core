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
    BudgetSnapshot,
    CompactionConfig,
    HarnessBuilder,
    HarnessConfig,
    HarnessRunOptions,
    InMemoryStorageProvider,
    LoopStrategyPlanExecute,
    LoopStrategyReAct,
    NoopContextManager,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    StandardCompactionAdapter,
    StandardContextManager,
    StandardHarness,
    StorageProvider,
    Task,
)
from spore_core.agent import Context, FinalResponse, ToolCallRequested, TurnResult
from spore_core.model import ModelParams, ProviderInfo, Role, TokenUsage, ToolCall


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


# ---------------------------------------------------------------------------
# #93: builder model_params reach every tool-requesting turn.
#
# ``_CapturingAgent`` records every ``Context`` it sees in ``seen``, and the
# agent copies ``Context.params`` verbatim into the ``ModelRequest``
# (``into_request``). So asserting on a captured context's
# ``params.structured_tool_calls`` proves the configured params reached the
# request the model would have seen.
# ---------------------------------------------------------------------------


class _ScriptedCapturingAgent:
    """Records every ``Context`` it is handed and yields scripted final
    responses on successive turns (so it can drive a PlanExecute run)."""

    def __init__(self, responses: list[str]) -> None:
        self.seen: list[Context] = []
        self._responses = list(responses)

    async def turn(self, context: Context) -> TurnResult:
        self.seen.append(context)
        text = self._responses.pop(0) if self._responses else "done"
        return FinalResponse(
            content=text,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    def id(self) -> AgentId:
        return AgentId("scripted-capture")


def _structured_params() -> ModelParams:
    return ModelParams(structured_tool_calls=True)


def _plan_execute_config(agent: object) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,  # type: ignore[arg-type]
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        storage=StorageProvider.single(InMemoryStorageProvider()),
        model_params=_structured_params(),
    )


async def test_default_model_params_are_default() -> None:
    """No ``.model_params(...)`` ⇒ each turn's context carries the default
    (``structured_tool_calls`` is False)."""
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
    task = Task.new("do a thing", SessionId("dflt"), LoopStrategyReAct(max_iterations=4))
    result = await harness.run(HarnessRunOptions(task))
    assert isinstance(result, RunResultSuccess)
    assert agent.seen, "agent should have seen at least one turn"
    assert not agent.seen[0].params.structured_tool_calls


async def test_model_params_reach_react_turn() -> None:
    """``.model_params(structured_tool_calls=True)`` ⇒ the ReAct turn context
    carries it."""
    agent = _CapturingAgent()
    harness = (
        HarnessBuilder(
            agent,
            ScriptedToolRegistry(),
            AllowAllSandbox(),
            _rich_adapter(),
            AlwaysContinuePolicy(),
        )
        .model_params(_structured_params())
        .build()
    )
    task = Task.new("do a thing", SessionId("react"), LoopStrategyReAct(max_iterations=4))
    result = await harness.run(HarnessRunOptions(task))
    assert isinstance(result, RunResultSuccess)
    assert agent.seen
    assert agent.seen[0].params.structured_tool_calls


async def test_model_params_reach_plan_phase() -> None:
    """The PlanExecute plan phase replaces params on its own seam — the
    plan-turn context carries the flag."""
    agent = _ScriptedCapturingAgent(['{"tasks":["one"],"rationale":"r"}'])
    harness = StandardHarness(_plan_execute_config(agent))
    task = Task.new("build something", SessionId("plan"), LoopStrategyPlanExecute(plan_model=None))
    state = SessionState()
    await harness._run_plan_phase(task, state, BudgetSnapshot(), None)
    assert len(agent.seen) == 1, "exactly one plan turn"
    assert agent.seen[0].params.structured_tool_calls


async def test_model_params_reach_execute_subloop() -> None:
    """A full PlanExecute run threads params through the shared react seam used
    by the execute sub-loop — every captured context carries the flag."""
    agent = _ScriptedCapturingAgent(
        [
            '{"tasks":["one","two"],"rationale":"r"}',
            "did one",
            "did two",
        ]
    )
    harness = StandardHarness(_plan_execute_config(agent))
    task = Task.new("build something", SessionId("exec"), LoopStrategyPlanExecute(plan_model=None))
    result = await harness.run(HarnessRunOptions(task))
    assert isinstance(result, RunResultSuccess)
    # 1 plan turn + 2 execute turns; every captured context carries it.
    assert len(agent.seen) == 3
    assert all(c.params.structured_tool_calls for c in agent.seen)


# ---------------------------------------------------------------------------
# #93: the plan-phase directive must NOT leak into the execute-step contexts.
#
# The plan phase ("Produce a step-by-step plan… Respond with a single JSON
# object…") used to be appended to the SHARED session_state, which the execute
# phase threads into every subtask sub-loop. An instruction-following model then
# re-emits a plan each step instead of calling tools. The fix scratch-clones the
# planning turn so the directive reaches only the plan turn. Mirrors Rust's
# ``plan_directive_does_not_leak_into_execute_context``.
# ---------------------------------------------------------------------------


_PLAN_DIRECTIVE_HEAD = "Produce a step-by-step plan"
_PLAN_DIRECTIVE_JSON = "Respond with a single JSON object"


class _ScriptedTurnAgent:
    """Records every ``Context`` it is handed and yields a scripted sequence of
    turn results (tool calls and/or finals) so it can drive a full PlanExecute
    run through both phases."""

    def __init__(self, results: list[TurnResult]) -> None:
        self.seen: list[Context] = []
        self._results = list(results)

    async def turn(self, context: Context) -> TurnResult:
        self.seen.append(context)
        if self._results:
            return self._results.pop(0)
        return FinalResponse(content="done", usage=TokenUsage(input_tokens=1, output_tokens=1))

    def id(self) -> AgentId:
        return AgentId("scripted-turn")


def _context_texts(context: Context) -> str:
    """Concatenate the text of every message in a captured context."""
    return "\n".join(getattr(m.content, "text", "") or "" for m in context.messages)


async def test_plan_directive_does_not_leak_into_execute_context() -> None:
    """A full 2-step PlanExecute run: the plan turn sees the directive, but no
    execute-step context carries it, and the steps receive their instructions."""
    agent = _ScriptedTurnAgent(
        [
            # Plan turn: produce a 2-step plan.
            FinalResponse(
                content='{"tasks":["step one","step two"],"rationale":"r"}',
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            ),
            # Execute step 1: issue a tool call, then finalize.
            ToolCallRequested(
                calls=[ToolCall(id="c1", name="noop")],
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            ),
            FinalResponse(
                content="did step one", usage=TokenUsage(input_tokens=1, output_tokens=1)
            ),
            # Execute step 2: finalize directly.
            FinalResponse(
                content="did step two", usage=TokenUsage(input_tokens=1, output_tokens=1)
            ),
        ]
    )
    harness = StandardHarness(
        HarnessConfig(
            agent=agent,
            tool_registry=ScriptedToolRegistry(),
            sandbox=AllowAllSandbox(),
            context_manager=_rich_adapter(),
            termination_policy=AlwaysContinuePolicy(),
            storage=StorageProvider.single(InMemoryStorageProvider()),
        )
    )
    task = Task.new(
        "build something",
        SessionId("leak"),
        LoopStrategyPlanExecute(plan_model=None),
    )
    result = await harness.run(HarnessRunOptions(task))
    assert isinstance(result, RunResultSuccess), f"expected Success, got {result!r}"

    # 1 plan turn + (tool call + final) for step one + 1 for step two.
    assert len(agent.seen) == 4, "plan turn + 3 execute turns"

    # The PLAN turn (index 0) DOES carry the directive — that's correct.
    plan_text = _context_texts(agent.seen[0])
    assert _PLAN_DIRECTIVE_HEAD in plan_text, f"plan turn should see the directive: {plan_text!r}"
    assert _PLAN_DIRECTIVE_JSON in plan_text, f"plan turn should see the directive: {plan_text!r}"

    # No EXECUTE-step context (indices 1..) may carry the directive.
    for i, c in enumerate(agent.seen[1:], start=1):
        text = _context_texts(c)
        assert _PLAN_DIRECTIVE_HEAD not in text, (
            f"execute-step context {i} leaked the directive: {text!r}"
        )
        assert _PLAN_DIRECTIVE_JSON not in text, (
            f"execute-step context {i} leaked the directive: {text!r}"
        )

    # The execute steps still receive their step instructions.
    assert "step one" in _context_texts(agent.seen[1]), (
        "step-one context should carry its instruction"
    )
    assert "step two" in _context_texts(agent.seen[3]), (
        "step-two context should carry its instruction"
    )


async def test_model_params_reach_streaming_turn() -> None:
    """The streaming path flows through ``_run_react_inner``'s same seam — the
    streamed turn's captured context carries the flag."""
    agent = _CapturingAgent()
    harness = (
        HarnessBuilder(
            agent,
            ScriptedToolRegistry(),
            AllowAllSandbox(),
            _rich_adapter(),
            AlwaysContinuePolicy(),
        )
        .model_params(_structured_params())
        .build()
    )
    task = Task.new("do a thing", SessionId("stream"), LoopStrategyReAct(max_iterations=4))
    result = await harness.run(HarnessRunOptions(task, on_stream=lambda _ev: None))
    assert isinstance(result, RunResultSuccess)
    assert agent.seen
    assert agent.seen[0].params.structured_tool_calls
