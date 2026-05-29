"""Tests for the PlanExecute plan phase on :class:`StandardHarness` (issue #70).

Mirrors the plan-phase unit tests in ``rust/crates/spore-core/src/harness.rs``.
Each test exercises one rule (R1-R11) or the Q4 terminal-halt decision; the rule
lives in the test docstring.
"""

from __future__ import annotations

from pathlib import Path

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetLimits,
    FinalResponse,
    HaltReasonAgentError,
    HaltReasonBudgetExceeded,
    HaltReasonExecutePhaseNotImplemented,
    HaltReasonPlanPhaseFailed,
    HarnessBuilder,
    HarnessConfig,
    HarnessRunOptions,
    LoopStrategyPlanExecute,
    MockAgent,
    ModelAgent,
    NoopContextManager,
    PLAN_EXECUTE_EXTRAS_KEY,
    PlanArtifact,
    ProviderInfo,
    ReplayModelInterface,
    RunResultFailure,
    ScriptedToolRegistry,
    SessionId,
    StandardHarness,
    Task,
    TokenUsage,
    ToolCall,
    ToolCallRequested,
)
from spore_core.hooks import (
    FunctionHook,
    HookContinue,
    HookEvent,
    OnPlanCreatedContext,
    StandardHookChain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent() -> MockAgent:
    return MockAgent(AgentId("test"))


def _config(agent: MockAgent, **overrides: object) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=overrides.get("tool_registry", ScriptedToolRegistry()),
        sandbox=overrides.get("sandbox", AllowAllSandbox()),
        context_manager=overrides.get("context_manager", NoopContextManager()),
        termination_policy=overrides.get("termination_policy", AlwaysContinuePolicy()),
        observability=overrides.get("observability"),
        hooks=overrides.get("hooks"),
        planner_agent=overrides.get("planner_agent"),
    )


def _plan_task(*, max_turns: int | None = None) -> Task:
    return Task.new(
        "build something",
        SessionId("plan-s1"),
        LoopStrategyPlanExecute(plan_model=None),
        budget=BudgetLimits(max_turns=max_turns),
    )


def _usage(in_t: int = 1, out_t: int = 1) -> TokenUsage:
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


_PLAN_JSON = '{"tasks":["a","b"],"rationale":"r"}'


# ---------------------------------------------------------------------------
# R1: the plan phase runs exactly once (one planner turn).
# ---------------------------------------------------------------------------


async def test_plan_phase_runs_exactly_once() -> None:
    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = StandardHarness(_config(a))
    r = await h.run(HarnessRunOptions(_plan_task()))
    assert isinstance(r, RunResultFailure)
    assert a.call_count == 1
    assert r.turns == 1


# ---------------------------------------------------------------------------
# R2: one-shot — a tool call in the plan turn is a PlanningTurnFailed, never a
# dispatch loop.
# ---------------------------------------------------------------------------


async def test_plan_turn_tool_call_is_planning_failure() -> None:
    a = _agent()
    a.push(ToolCallRequested(calls=[ToolCall(id="c", name="x", input={})], usage=_usage()))
    from spore_core import SessionState

    reg = ScriptedToolRegistry()
    h = StandardHarness(_config(a, tool_registry=reg))
    state = SessionState()
    r = await h.run(HarnessRunOptions(_plan_task(), session_state=state))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonPlanPhaseFailed)
    # Error nested under `error` (3-language parity), not flattened.
    assert r.reason.error.kind == "planning_turn_failed"
    assert reg.call_count == 0  # never dispatched
    assert PLAN_EXECUTE_EXTRAS_KEY not in state.extras  # no artifact stored


# ---------------------------------------------------------------------------
# R3: the artifact is captured from the response text.
# R4: the artifact is stored in extras["plan_execute"] as a JSON object.
# ---------------------------------------------------------------------------


async def test_artifact_captured_and_stored_in_extras() -> None:
    from spore_core import SessionState

    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = StandardHarness(_config(a))
    state = SessionState()
    r = await h.run(HarnessRunOptions(_plan_task(), session_state=state))
    assert isinstance(r, RunResultFailure)
    stored = state.extras[PLAN_EXECUTE_EXTRAS_KEY]
    # Stored as a JSON-safe object (matches Rust's serde_json::to_value).
    assert stored == {"tasks": ["a", "b"], "rationale": "r"}


# ---------------------------------------------------------------------------
# R3 (unparseable): a bad response surfaces PlanPhaseFailed/unparseable and
# stores no artifact.
# ---------------------------------------------------------------------------


async def test_unparseable_plan_fails_and_stores_nothing() -> None:
    from spore_core import SessionState

    a = _agent()
    a.push(FinalResponse(content="not json", usage=_usage()))
    h = StandardHarness(_config(a))
    state = SessionState()
    r = await h.run(HarnessRunOptions(_plan_task(), session_state=state))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonPlanPhaseFailed)
    assert r.reason.error.kind == "unparseable_plan"
    assert PLAN_EXECUTE_EXTRAS_KEY not in state.extras
    # Wire shape: the error is NESTED under `error` (parity with Rust's
    # HaltReason::PlanPhaseFailed { error } and Go's HaltPlanPhaseFailed), not
    # flattened as top-level error_kind/message fields.
    dumped = r.reason.model_dump(mode="json")
    assert dumped["kind"] == "plan_phase_failed"
    assert set(dumped) == {"kind", "error"}
    assert dumped["error"]["kind"] == "unparseable_plan"
    assert "message" in dumped["error"]
    assert "error_kind" not in dumped


# ---------------------------------------------------------------------------
# Agent error in the plan turn surfaces AgentError, stores nothing.
# ---------------------------------------------------------------------------


async def test_plan_turn_agent_error() -> None:
    from spore_core import SessionState

    a = _agent()  # empty MockAgent → returns AgentErrorEmpty
    h = StandardHarness(_config(a))
    state = SessionState()
    r = await h.run(HarnessRunOptions(_plan_task(), session_state=state))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonAgentError)
    assert PLAN_EXECUTE_EXTRAS_KEY not in state.extras


# ---------------------------------------------------------------------------
# R5: when planner_agent is set, the PLANNER runs the plan turn and the default
# agent does not.
# ---------------------------------------------------------------------------


async def test_plan_phase_routes_to_planner_agent() -> None:
    default = _agent()
    default.push(FinalResponse(content='{"tasks":["default ran"]}', usage=_usage()))
    planner = _agent()
    planner.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = StandardHarness(_config(default, planner_agent=planner))
    r = await h.run(HarnessRunOptions(_plan_task()))
    assert isinstance(r, RunResultFailure)
    assert planner.call_count == 1
    assert default.call_count == 0


# ---------------------------------------------------------------------------
# R6: with no planner_agent, the plan turn runs on the default agent.
# ---------------------------------------------------------------------------


async def test_plan_phase_uses_default_agent_without_planner() -> None:
    default = _agent()
    default.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = StandardHarness(_config(default))
    r = await h.run(HarnessRunOptions(_plan_task()))
    assert isinstance(r, RunResultFailure)
    assert default.call_count == 1


# ---------------------------------------------------------------------------
# R7: the plan turn counts against the shared budget.
# ---------------------------------------------------------------------------


async def test_plan_turn_counts_against_budget() -> None:
    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage(in_t=4, out_t=2)))
    h = StandardHarness(_config(a))
    r = await h.run(HarnessRunOptions(_plan_task(max_turns=5)))
    assert isinstance(r, RunResultFailure)
    assert r.turns == 1
    assert r.usage.input_tokens == 4
    assert r.usage.output_tokens == 2


# ---------------------------------------------------------------------------
# R8: exactly one turn span is recorded for the plan turn.
# ---------------------------------------------------------------------------


async def test_one_turn_span_recorded() -> None:
    from spore_core import InMemoryObservabilityProvider

    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    obs = InMemoryObservabilityProvider()
    h = StandardHarness(_config(a, observability=obs))
    task = _plan_task()
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    metrics = await obs.get_session_metrics(task.session_id)
    assert metrics is not None
    assert metrics.total_turns == 1


# ---------------------------------------------------------------------------
# R10: budget exhausted before the plan turn → budget-exceeded failure, no
# artifact, and the planner never runs.
# ---------------------------------------------------------------------------


async def test_budget_exhausted_before_plan_turn() -> None:
    from spore_core import SessionState

    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = StandardHarness(_config(a))
    # max_turns=0 means the budget is already exhausted before the plan turn.
    task = Task.new(
        "build something",
        SessionId("plan-s1"),
        LoopStrategyPlanExecute(plan_model=None),
        budget=BudgetLimits(max_turns=0),
    )
    state = SessionState()
    r = await h.run(HarnessRunOptions(task, session_state=state))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonBudgetExceeded)
    assert r.reason.limit_type == "turns"
    assert a.call_count == 0
    assert PLAN_EXECUTE_EXTRAS_KEY not in state.extras


# ---------------------------------------------------------------------------
# R11: an OnPlanCreated hook can rewrite the plan before storage; the stored
# artifact reflects the mutation.
# ---------------------------------------------------------------------------


async def test_on_plan_created_mutation_reflected_in_stored_artifact() -> None:
    # Mirrors Rust's ``deferred_on_plan_created_mutates``: the hook mutates the
    # plan IN PLACE inside a Continue-returning handler (OnPlanCreated is a
    # post-event, so the HookMutate decision is not how it rewrites — direct
    # mutation of the mutable ``plan`` field is).
    from spore_core import SessionState

    def handler(ctx: object) -> HookContinue:
        assert isinstance(ctx, OnPlanCreatedContext)
        ctx.plan.tasks.append("extra")
        ctx.plan.rationale = "rewritten"
        return HookContinue()

    chain = StandardHookChain()
    chain.register(FunctionHook("rewrite", [HookEvent.ON_PLAN_CREATED], handler))
    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = StandardHarness(_config(a, hooks=chain))
    state = SessionState()
    r = await h.run(HarnessRunOptions(_plan_task(), session_state=state))
    assert isinstance(r, RunResultFailure)
    assert state.extras[PLAN_EXECUTE_EXTRAS_KEY] == {
        "tasks": ["a", "b", "extra"],
        "rationale": "rewritten",
    }


# A non-mutating (Continue) OnPlanCreated hook leaves the captured plan intact.
async def test_on_plan_created_continue_keeps_captured_plan() -> None:
    from spore_core import SessionState

    seen: list[PlanArtifact] = []

    def handler(ctx: object) -> HookContinue:
        assert isinstance(ctx, OnPlanCreatedContext)
        seen.append(ctx.plan)
        return HookContinue()

    chain = StandardHookChain()
    chain.register(FunctionHook("observe", [HookEvent.ON_PLAN_CREATED], handler))
    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = StandardHarness(_config(a, hooks=chain))
    state = SessionState()
    await h.run(HarnessRunOptions(_plan_task(), session_state=state))
    assert len(seen) == 1
    assert seen[0].tasks == ["a", "b"]
    assert state.extras[PLAN_EXECUTE_EXTRAS_KEY] == {"tasks": ["a", "b"], "rationale": "r"}


# ---------------------------------------------------------------------------
# Q4: after producing+storing an artifact, the full PlanExecute run() halts with
# the distinct ExecutePhaseNotImplemented reason (not StrategyNotYetImplemented).
# ---------------------------------------------------------------------------


async def test_plan_execute_halts_execute_phase_not_implemented() -> None:
    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = StandardHarness(_config(a))
    r = await h.run(HarnessRunOptions(_plan_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonExecutePhaseNotImplemented)


# ---------------------------------------------------------------------------
# Builder wires planner_agent.
# ---------------------------------------------------------------------------


async def test_builder_planner_agent_setter() -> None:
    default = _agent()
    planner = _agent()
    planner.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = (
        HarnessBuilder(
            default,
            ScriptedToolRegistry(),
            AllowAllSandbox(),
            NoopContextManager(),
            AlwaysContinuePolicy(),
        )
        .planner_agent(planner)
        .build()
    )
    r = await h.run(HarnessRunOptions(_plan_task()))
    assert isinstance(r, RunResultFailure)
    assert planner.call_count == 1
    assert default.call_count == 0


# ---------------------------------------------------------------------------
# Fixture-replay test (parity with Rust plan_phase_fixture_replay.rs).
# ---------------------------------------------------------------------------


def _fixture_path() -> Path:
    # tests/  →  spore_core/  →  packages/  →  python/  →  repo-root
    here = Path(__file__).resolve()
    return here.parents[4] / "fixtures" / "model_responses" / "harness" / "plan_phase_basic.jsonl"


def _fixture_responses() -> list[str]:
    """Extract the text block from each recorded response in the fixture."""
    import json as _json

    from pydantic import TypeAdapter

    from spore_core.model import ModelResponse, TextBlock

    texts: list[str] = []
    adapter = TypeAdapter(ModelResponse)
    for line in _fixture_path().read_text().splitlines():
        if not line.strip():
            continue
        row = _json.loads(line)
        resp = adapter.validate_python(row["response"])
        block = resp.content[0]
        assert isinstance(block, TextBlock)
        texts.append(block.text)
    return texts


async def _drive_plan_phase(response_text: str) -> RunResultFailure:
    """Drive the full harness plan phase against a single replayed response and
    assert it halts with ExecutePhaseNotImplemented (proving the harness
    consumed the replayed planner response)."""
    from spore_core import SessionState
    from spore_core.model import (
        ModelRequest,
        ModelResponse,
        RecordedExchange,
        StopReason,
        TextBlock,
    )
    from spore_core.model import TokenUsage as MTokenUsage

    # Build a single-exchange positional replay returning ``response_text``.
    exchange = RecordedExchange(
        request=ModelRequest(),
        response=ModelResponse(
            content=[TextBlock(text=response_text)],
            usage=MTokenUsage(input_tokens=1, output_tokens=1),
            stop_reason=StopReason.END_TURN,
        ),
        provider="anthropic",
    )
    replay = ReplayModelInterface(
        [exchange],
        ProviderInfo(name="anthropic", model_id="fixture", context_window=200_000),
    )
    agent = ModelAgent(AgentId("planner"), replay)
    h = StandardHarness(_config_for_agent(agent))
    state = SessionState()
    r = await h.run(
        HarnessRunOptions(
            Task.new(
                "build something",
                SessionId("plan-fixture"),
                LoopStrategyPlanExecute(plan_model=None),
            ),
            session_state=state,
        )
    )
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonExecutePhaseNotImplemented)
    assert r.turns == 1
    # Stash the extras on the result object is not possible; assert directly.
    _LAST_EXTRAS.clear()
    _LAST_EXTRAS.update(state.extras)
    return r


_LAST_EXTRAS: dict[str, object] = {}


def _config_for_agent(agent: ModelAgent) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
    )


async def test_fixture_plain_json_captures_exact_artifact() -> None:
    texts = _fixture_responses()
    assert len(texts) >= 2
    await _drive_plan_phase(texts[0])
    assert _LAST_EXTRAS[PLAN_EXECUTE_EXTRAS_KEY] == {
        "tasks": [
            "scaffold the project",
            "add the argument parser",
            "write the integration tests",
        ],
        "rationale": "deliver a working CLI incrementally",
    }


async def test_fixture_fenced_json_captures_exact_artifact() -> None:
    texts = _fixture_responses()
    assert len(texts) >= 2
    await _drive_plan_phase(texts[1])
    assert _LAST_EXTRAS[PLAN_EXECUTE_EXTRAS_KEY] == {
        "tasks": ["draft the outline", "write the reference section"],
        "rationale": "docs follow the code",
    }
