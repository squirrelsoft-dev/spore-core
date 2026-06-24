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
    HarnessConfig,
    HarnessRunOptions,
    InMemoryStorageProvider,
    PlanExecuteConfig,
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
    StorageProvider,
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
from spore_core.storage import project_namespace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent() -> MockAgent:
    return MockAgent(AgentId("test"))


def _config(agent: MockAgent, **overrides: object) -> HarnessConfig:
    # #76: the plan artifact now lives on the RunStore seam (not
    # SessionState.extras), so the test harness needs a real (in-memory) run
    # store for the readback assertions below to observe what the harness wrote.
    return HarnessConfig(
        agent=agent,
        tool_registry=overrides.get("tool_registry", ScriptedToolRegistry()),
        sandbox=overrides.get("sandbox", AllowAllSandbox()),
        context_manager=overrides.get("context_manager", NoopContextManager()),
        termination_policy=overrides.get("termination_policy", AlwaysContinuePolicy()),
        observability=overrides.get("observability"),
        hooks=overrides.get("hooks"),
        planner_agent=overrides.get("planner_agent"),
        storage=overrides.get("storage", StorageProvider.single(InMemoryStorageProvider())),
    )


async def _stored_artifact(h: StandardHarness, session_id: SessionId) -> object:
    """Read the plan artifact back through the harness's RunStore seam (#76).
    #142: the plan artifact is keyed by the project namespace, not the run
    session id — read it back under ``project_namespace(h.project_id())``."""
    _ = session_id  # #142: durable readback keys by the project namespace.
    return await h.storage().run().get(project_namespace(h.project_id()), PLAN_EXECUTE_EXTRAS_KEY)


def _plan_task(*, max_turns: int | None = None) -> Task:
    return Task.new(
        "build something",
        SessionId("plan-s1"),
        PlanExecuteConfig.simple(),
        budget=BudgetLimits(max_turns=max_turns),
    )


def _usage(in_t: int = 1, out_t: int = 1) -> TokenUsage:
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


_PLAN_JSON = '{"tasks":["a","b"],"rationale":"r"}'


# ---------------------------------------------------------------------------
# R1: the plan phase runs exactly once (one planner turn).
# ---------------------------------------------------------------------------


async def test_plan_phase_runs_exactly_once() -> None:
    from spore_core import BudgetSnapshot, SessionState
    from spore_core.harness import _PlanPhaseOutcome

    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = StandardHarness(_config(a))
    state = SessionState()
    outcome = await h._run_plan_phase(_plan_task(), state, BudgetSnapshot(), None)
    assert isinstance(outcome, _PlanPhaseOutcome)
    assert a.call_count == 1
    assert outcome.turns == 1


# ---------------------------------------------------------------------------
# #124 recursion seam (was R2): the plan phase now dispatches the genuine
# ``self.plan`` child (a ReAct loop) capped at ONE turn. A plan turn that requests
# a tool instead of emitting the JSON plan cannot complete in its single turn, so
# the plan child halts (budget) and the plan phase propagates a terminal Failure
# WITHOUT capturing/storing an artifact. (The old one-shot primitive special-cased
# a tool call as ``planning_turn_failed``; under genuine recursion the cap is what
# stops the loop — the observable contract that a non-plan plan turn fails the run
# and stores nothing is preserved.) Mirrors Rust's
# ``plan_phase_tool_call_fails_and_stores_no_artifact``.
# ---------------------------------------------------------------------------


async def test_plan_turn_tool_call_fails_and_stores_no_artifact() -> None:
    from spore_core import SessionState, ToolOutputSuccess

    a = _agent()
    a.push(ToolCallRequested(calls=[ToolCall(id="c", name="x", input={})], usage=_usage()))
    # The plan child is a genuine ReAct loop: its one turn dispatches the tool
    # before the one-turn cap halts the loop, so the registry serves one output.
    reg = ScriptedToolRegistry()
    reg.push(ToolOutputSuccess(content="ok", truncated=False))
    h = StandardHarness(_config(a, tool_registry=reg))
    state = SessionState()
    task = _plan_task()
    r = await h.run(HarnessRunOptions(task, session_state=state))
    # The plan child never emitted a JSON plan, so the run halts terminally.
    assert isinstance(r, RunResultFailure)
    # Nothing captured/stored: no artifact reached the RunStore.
    assert await _stored_artifact(h, task.session_id) is None
    assert PLAN_EXECUTE_EXTRAS_KEY not in state.extras  # never mirrored into extras


# ---------------------------------------------------------------------------
# R3: the artifact is captured from the response text.
# R4: the artifact is persisted to the RunStore under PLAN_EXECUTE_EXTRAS_KEY
#     as a JSON object (#76 — no longer mirrored into extras).
# ---------------------------------------------------------------------------


async def test_artifact_captured_and_stored_in_run_store() -> None:
    from spore_core import BudgetSnapshot, SessionState
    from spore_core.harness import _PlanPhaseOutcome

    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = StandardHarness(_config(a))
    state = SessionState()
    task = _plan_task()
    outcome = await h._run_plan_phase(task, state, BudgetSnapshot(), None)
    assert isinstance(outcome, _PlanPhaseOutcome)
    stored = await _stored_artifact(h, task.session_id)
    # Stored as a JSON-safe object (matches Rust's serde_json::to_value).
    assert stored == {"tasks": ["a", "b"], "rationale": "r"}
    # #76: not mirrored into SessionState.extras anymore.
    assert PLAN_EXECUTE_EXTRAS_KEY not in state.extras


# ---------------------------------------------------------------------------
# SC-28 acceptance (1): a free-text / markdown plan no longer fails the plan
# phase. The strict JSON grammar (+ prose repair) misses, so the driver captures
# it as a prose artifact instead of aborting: ``rationale`` holds the verbatim
# prose and ``tasks`` is sourced from the ``task_list`` tool store (empty here —
# the planner authored none). The artifact IS stored (R4 still applies) and
# round-trips (acceptance 3). Mirrors Rust's
# ``plan_phase_freetext_response_captures_as_prose`` (was
# ``plan_phase_unparseable_response_fails``).
# ---------------------------------------------------------------------------


async def test_plan_phase_freetext_response_captures_as_prose() -> None:
    from spore_core import BudgetSnapshot, SessionState
    from spore_core.harness import _PlanPhaseOutcome

    prose = "This is a markdown plan.\n\n1. do the thing\n2. do the other"
    a = _agent()
    a.push(FinalResponse(content=prose, usage=_usage()))
    h = StandardHarness(_config(a))
    state = SessionState()
    task = _plan_task()
    outcome = await h._run_plan_phase(task, state, BudgetSnapshot(), None)
    # A free-text plan captures rather than failing.
    assert isinstance(outcome, _PlanPhaseOutcome)
    # Prose preserved verbatim; no JSON tasks parsed and no task_list authored.
    assert outcome.artifact.rationale == prose
    assert outcome.artifact.tasks == []
    # R4: the artifact IS stored now (was absent pre-SC-28) and round-trips.
    stored = await _stored_artifact(h, task.session_id)
    assert stored == {"tasks": [], "rationale": prose}
    # #76: not mirrored into SessionState.extras.
    assert PLAN_EXECUTE_EXTRAS_KEY not in state.extras


# ---------------------------------------------------------------------------
# SC-28 acceptance (2): a markdown plan captures without parse failure AND the
# OnPlanCreated payload's ``tasks`` are sourced from the ``task_list`` tool store
# (the ONE authoring path) — so panel consumers (looper ``plan_tracker``,
# cordyceps ``plan_announcer``) still get task texts even though the plan prose is
# free-text rather than the JSON ``PlanArtifact``. Mirrors Rust's
# ``plan_phase_freetext_sources_tasks_from_task_list``.
#
# NOTE: a REAL in-memory RunStore is wired (``_config`` uses
# ``InMemoryStorageProvider``), NOT the default no-op store — otherwise the
# seeded ``task_list`` would silently vanish and the test would assert the empty
# path while passing.
# ---------------------------------------------------------------------------


async def test_plan_phase_freetext_sources_tasks_from_task_list() -> None:
    from spore_core import BudgetSnapshot, SessionState
    from spore_core.harness import _PlanPhaseOutcome
    from spore_core.tasklist import TASK_LIST_EXTRAS_KEY, TaskList

    prose = "Here's my plan in prose, no JSON object at all."
    a = _agent()
    a.push(FinalResponse(content=prose, usage=_usage()))
    h = StandardHarness(_config(a))
    task = _plan_task()

    # Seed the durable task_list store as if the plan leaf had authored it via the
    # ``task_list`` tool during the plan turn (keyed by the project namespace).
    seeded = TaskList()
    seeded.add("build it", [])
    seeded.add("test it", [])
    await (
        h.storage()
        .run()
        .put(
            project_namespace(h.project_id()),
            TASK_LIST_EXTRAS_KEY,
            seeded.to_dict(),
        )
    )

    state = SessionState()
    outcome = await h._run_plan_phase(task, state, BudgetSnapshot(), None)
    assert isinstance(outcome, _PlanPhaseOutcome)
    # ``tasks`` pulled from the seeded task_list; prose preserved as rationale.
    assert outcome.artifact.tasks == ["build it", "test it"]
    assert outcome.artifact.rationale == prose
    # Stored artifact round-trips (acceptance 3).
    stored = await _stored_artifact(h, task.session_id)
    assert stored == {"tasks": ["build it", "test it"], "rationale": prose}


# ---------------------------------------------------------------------------
# Agent error in the plan turn surfaces AgentError, stores nothing.
# ---------------------------------------------------------------------------


async def test_plan_turn_agent_error() -> None:
    from spore_core import SessionState

    a = _agent()  # empty MockAgent → returns AgentErrorEmpty
    h = StandardHarness(_config(a))
    state = SessionState()
    task = _plan_task()
    r = await h.run(HarnessRunOptions(task, session_state=state))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonAgentError)
    assert await _stored_artifact(h, task.session_id) is None
    assert PLAN_EXECUTE_EXTRAS_KEY not in state.extras


# ---------------------------------------------------------------------------
# #124 Q1: the separate ``planner_agent`` concept is DROPPED — the plan child's
# leaf ``ReactConfig.agent`` is authoritative (the recursing leaf resolves it).
# The former ``test_plan_phase_routes_to_planner_agent`` is removed accordingly.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# R6: the plan turn runs on the resolved (default) worker agent.
# ---------------------------------------------------------------------------


async def test_plan_phase_uses_default_agent_without_planner() -> None:
    from spore_core import BudgetSnapshot, SessionState
    from spore_core.harness import _PlanPhaseOutcome

    default = _agent()
    default.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = StandardHarness(_config(default))
    outcome = await h._run_plan_phase(_plan_task(), SessionState(), BudgetSnapshot(), None)
    assert isinstance(outcome, _PlanPhaseOutcome)
    assert default.call_count == 1


# ---------------------------------------------------------------------------
# R7: the plan turn counts against the shared budget.
# ---------------------------------------------------------------------------


async def test_plan_turn_counts_against_budget() -> None:
    from spore_core import BudgetSnapshot, SessionState
    from spore_core.harness import _PlanPhaseOutcome

    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage(in_t=4, out_t=2)))
    h = StandardHarness(_config(a))
    outcome = await h._run_plan_phase(
        _plan_task(max_turns=5), SessionState(), BudgetSnapshot(), None
    )
    assert isinstance(outcome, _PlanPhaseOutcome)
    assert outcome.turns == 1
    assert outcome.usage.input_tokens == 4
    assert outcome.usage.output_tokens == 2


# ---------------------------------------------------------------------------
# R8: exactly one turn span is recorded for the plan turn.
# ---------------------------------------------------------------------------


async def test_one_turn_span_recorded() -> None:
    from spore_core import BudgetSnapshot, InMemoryObservabilityProvider, SessionState
    from spore_core.harness import _PlanPhaseOutcome

    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    obs = InMemoryObservabilityProvider()
    h = StandardHarness(_config(a, observability=obs))
    task = _plan_task()
    outcome = await h._run_plan_phase(task, SessionState(), BudgetSnapshot(), None)
    assert isinstance(outcome, _PlanPhaseOutcome)
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
        PlanExecuteConfig.simple(),
        budget=BudgetLimits(max_turns=0),
    )
    state = SessionState()
    r = await h.run(HarnessRunOptions(task, session_state=state))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonBudgetExceeded)
    assert r.reason.limit_type == "turns"
    assert a.call_count == 0
    assert await _stored_artifact(h, task.session_id) is None
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

    from spore_core import BudgetSnapshot
    from spore_core.harness import _PlanPhaseOutcome

    chain = StandardHookChain()
    chain.register(FunctionHook("rewrite", [HookEvent.ON_PLAN_CREATED], handler))
    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = StandardHarness(_config(a, hooks=chain))
    state = SessionState()
    task = _plan_task()
    outcome = await h._run_plan_phase(task, state, BudgetSnapshot(), None)
    assert isinstance(outcome, _PlanPhaseOutcome)
    assert await _stored_artifact(h, task.session_id) == {
        "tasks": ["a", "b", "extra"],
        "rationale": "rewritten",
    }
    assert PLAN_EXECUTE_EXTRAS_KEY not in state.extras


# A non-mutating (Continue) OnPlanCreated hook leaves the captured plan intact.
async def test_on_plan_created_continue_keeps_captured_plan() -> None:
    from spore_core import SessionState

    seen: list[PlanArtifact] = []

    def handler(ctx: object) -> HookContinue:
        assert isinstance(ctx, OnPlanCreatedContext)
        seen.append(ctx.plan)
        return HookContinue()

    from spore_core import BudgetSnapshot

    chain = StandardHookChain()
    chain.register(FunctionHook("observe", [HookEvent.ON_PLAN_CREATED], handler))
    a = _agent()
    a.push(FinalResponse(content=_PLAN_JSON, usage=_usage()))
    h = StandardHarness(_config(a, hooks=chain))
    state = SessionState()
    task = _plan_task()
    await h._run_plan_phase(task, state, BudgetSnapshot(), None)
    assert len(seen) == 1
    assert seen[0].tasks == ["a", "b"]
    assert await _stored_artifact(h, task.session_id) == {"tasks": ["a", "b"], "rationale": "r"}
    assert PLAN_EXECUTE_EXTRAS_KEY not in state.extras


# ---------------------------------------------------------------------------
# The execute phase is now implemented (#59): ExecutePhaseNotImplemented is gone.
# ---------------------------------------------------------------------------


async def test_execute_phase_not_implemented_removed() -> None:
    import spore_core

    assert not hasattr(spore_core, "HaltReasonExecutePhaseNotImplemented")
    # The discriminated HaltReason union no longer accepts the old tag.
    import pytest
    from pydantic import TypeAdapter

    from spore_core.harness import HaltReason

    with pytest.raises(Exception):
        TypeAdapter(HaltReason).validate_python({"kind": "execute_phase_not_implemented"})


# ---------------------------------------------------------------------------
# #124 Q1: the ``planner_agent`` concept is DROPPED — the former
# ``test_builder_planner_agent_setter`` is removed accordingly.
# ---------------------------------------------------------------------------


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


async def _drive_plan_phase(response_text: str) -> None:
    """Drive the one-shot plan phase against a single replayed response and assert
    it produces+stores the artifact (proving the harness consumed the replayed
    planner response)."""
    from spore_core import BudgetSnapshot, SessionState
    from spore_core.harness import _PlanPhaseOutcome
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
    session_id = SessionId("plan-fixture")
    outcome = await h._run_plan_phase(
        Task.new(
            "build something",
            session_id,
            PlanExecuteConfig.simple(),
        ),
        state,
        BudgetSnapshot(),
        None,
    )
    assert isinstance(outcome, _PlanPhaseOutcome)
    assert outcome.turns == 1
    # #76: the artifact is read back from the RunStore seam, not extras.
    _LAST_ARTIFACT.clear()
    _LAST_ARTIFACT["value"] = await _stored_artifact(h, session_id)


_LAST_ARTIFACT: dict[str, object] = {}


def _config_for_agent(agent: ModelAgent) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        storage=StorageProvider.single(InMemoryStorageProvider()),
    )


async def test_fixture_plain_json_captures_exact_artifact() -> None:
    texts = _fixture_responses()
    assert len(texts) >= 2
    await _drive_plan_phase(texts[0])
    assert _LAST_ARTIFACT["value"] == {
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
    assert _LAST_ARTIFACT["value"] == {
        "tasks": ["draft the outline", "write the reference section"],
        "rationale": "docs follow the code",
    }
