"""Tests for ``HumanRequest::BudgetExhausted`` + Escalate HITL resume (issue #130).

Wires the ``Escalate`` budget behavior into the existing HITL pause/resume seam
(``RunResultWaitingForHuman``). Mirrors the ``#130`` tests in
``rust/crates/spore-core/src/harness.rs`` and the shared fixture-replay test.

Spec forks (RESOLVED by the maintainer — the Rust reference follows them):

* A — ``EscalationActionContinueWithBudget`` carries a NAMED ``steps`` field
  (kind-tagged, snake_case).
* B — a typed ``HumanResponseEscalate { action }`` delivers the operator's
  choice on resume; ``Allow`` / ``Halt`` / ``Deny`` are NOT overloaded.
* C — only combinators offer ``Skip`` in ``available_actions``; a bare leaf
  OMITS it.
* D — ``available_actions`` is ADVISORY for v1: resume does NOT hard-reject an
  out-of-set action.
* E — the node's budget context is reconstructed from the ``BudgetExhausted``
  request payload (``steps_taken`` + ``continues_used``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetLimits,
    BudgetSnapshot,
    EscalationAction,
    EscalationActionContinueWithBudget,
    EscalationActionFail,
    EscalationActionSkip,
    EscalationModeAutonomous,
    EscalationModeSurfaceToHuman,
    FinalResponse,
    HaltReasonBudgetExceeded,
    HarnessConfig,
    HarnessRunOptions,
    HumanRequest,
    HumanRequestBudgetExhausted,
    HumanRequestReview,
    HumanRequestToolApproval,
    HumanResponse,
    HumanResponseAllow,
    HumanResponseEscalate,
    HumanResponseHalt,
    InMemoryStorageProvider,
    MockAgent,
    NoopContextManager,
    PausedState,
    PlanExecuteConfig,
    ReactConfig,
    RunResultFailure,
    RunResultSuccess,
    RunResultWaitingForHuman,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    StandardHarness,
    StorageProvider,
    Task,
    ToolOutputSuccess,
    TokenUsage,
)
from spore_core.agent import ToolCallRequested
from spore_core.harness import (
    BudgetExhausted,
    BudgetExhaustedEscalate,
    BudgetPolicyTotalSteps,
    _combinator_escalation_actions,
    _grant_task_budget,
    _leaf_escalation_actions,
    _react_partial_json,
)
from spore_core.model import ToolCall

_HUMAN_REQUEST_ADAPTER = TypeAdapter(HumanRequest)
_HUMAN_RESPONSE_ADAPTER = TypeAdapter(HumanResponse)
_ESCALATION_ACTION_ADAPTER = TypeAdapter(EscalationAction)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(
    agent: MockAgent,
    tool_registry: ScriptedToolRegistry | None = None,
    *,
    escalation_mode: object = None,
) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=tool_registry if tool_registry is not None else ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        storage=StorageProvider.single(InMemoryStorageProvider()),
        escalation_mode=escalation_mode,  # type: ignore[arg-type]
    )


def _fixture_path() -> Path:
    return (
        Path(__file__).resolve().parents[4] / "fixtures" / "paused_states" / "budget_exhausted.json"
    )


# ===========================================================================
# Fork A / B — serde round-trips of the three new variants (kind-tagged,
# snake_case wire).
# ===========================================================================


def test_escalation_action_round_trips_every_variant() -> None:
    for v in (
        EscalationActionContinueWithBudget(steps=7),
        EscalationActionSkip(),
        EscalationActionFail(),
    ):
        json_s = v.model_dump_json()
        back = _ESCALATION_ACTION_ADAPTER.validate_json(json_s)
        assert back == v
    # Fork A: the named-field variant uses the kind-tagged form (no tuple).
    assert (
        EscalationActionContinueWithBudget(steps=7).model_dump_json()
        == '{"kind":"continue_with_budget","steps":7}'
    )
    assert EscalationActionSkip().model_dump_json() == '{"kind":"skip"}'
    assert EscalationActionFail().model_dump_json() == '{"kind":"fail"}'


def test_human_request_budget_exhausted_round_trips() -> None:
    req = HumanRequestBudgetExhausted(
        phase="plan_execute",
        policy=BudgetPolicyTotalSteps(value=6),
        steps_taken=5,
        continues_used=2,
        partial_output='{"node":"plan_execute"}',
        available_actions=[
            EscalationActionContinueWithBudget(steps=5),
            EscalationActionSkip(),
            EscalationActionFail(),
        ],
    )
    json_s = req.model_dump_json()
    back = _HUMAN_REQUEST_ADAPTER.validate_json(json_s)
    assert back == req
    # The variant tag is snake_case.
    assert '"kind":"budget_exhausted"' in json_s


def test_human_response_escalate_round_trips() -> None:
    r = HumanResponseEscalate(action=EscalationActionContinueWithBudget(steps=3))
    json_s = r.model_dump_json()
    back = _HUMAN_RESPONSE_ADAPTER.validate_json(json_s)
    assert back == r
    assert '"kind":"escalate"' in json_s


def test_partial_output_serialized_as_null_when_absent() -> None:
    # Fork: ``partial_output`` mirrors the Rust ``Option<String>`` with NO
    # skip-serializing — it is serialized as ``null``, NOT omitted.
    req = HumanRequestBudgetExhausted(
        phase="react",
        policy=BudgetPolicyTotalSteps(value=1),
        steps_taken=1,
        continues_used=0,
        partial_output=None,
        available_actions=[],
    )
    assert '"partial_output":null' in req.model_dump_json()


# ===========================================================================
# Existing-variant regression — the pre-#130 HumanRequest / HumanResponse
# variants are UNCHANGED and still round-trip.
# ===========================================================================


def test_existing_human_request_variants_unchanged() -> None:
    for req in (
        HumanRequestToolApproval(calls=[ToolCall(id="c0", name="x", input={})], risk_level="high"),
        HumanRequestReview(content="ship it?"),
    ):
        back = _HUMAN_REQUEST_ADAPTER.validate_json(req.model_dump_json())
        assert back == req


def test_existing_human_response_variants_unchanged() -> None:
    for r in (HumanResponseAllow(), HumanResponseHalt()):
        back = _HUMAN_RESPONSE_ADAPTER.validate_json(r.model_dump_json())
        assert back == r


# ===========================================================================
# escalation_mode() accessor.
# ===========================================================================


def test_escalation_mode_accessor_returns_config_mode() -> None:
    a = MockAgent(AgentId("test"))
    h_surface = StandardHarness(_config(a, escalation_mode=EscalationModeSurfaceToHuman()))
    assert isinstance(h_surface.escalation_mode(), EscalationModeSurfaceToHuman)
    h_afk = StandardHarness(_config(a, escalation_mode=EscalationModeAutonomous()))
    assert isinstance(h_afk.escalation_mode(), EscalationModeAutonomous)


# ===========================================================================
# Fork C — combinators offer Skip; bare leaf omits Skip.
# ===========================================================================


def test_action_sets_combinator_vs_leaf() -> None:
    err = BudgetExhausted(
        policy=BudgetPolicyTotalSteps(value=4),
        behavior=BudgetExhaustedEscalate(),
        steps_taken=4,
        continues_used=0,
        phase="plan_execute",
    )
    combinator = _combinator_escalation_actions(err)
    assert [a.kind for a in combinator] == ["continue_with_budget", "skip", "fail"]
    leaf = _leaf_escalation_actions(err)
    assert [a.kind for a in leaf] == ["continue_with_budget", "fail"]
    # ContinueWithBudget seeds ``steps`` from the spent allowance.
    assert isinstance(combinator[0], EscalationActionContinueWithBudget)
    assert combinator[0].steps == 4


# ===========================================================================
# Helpers to drive a REAL leaf / combinator budget exhaustion.
# ===========================================================================


async def _run_leaf_exhaustion(escalation_mode: object) -> object:
    """A ReAct leaf whose OWN ``PerLoop(2)`` cap is the binding constraint:
    3 tool-call turns, cap 2, no global cap."""
    a = MockAgent(AgentId("test"))
    for i in range(3):
        a.push(
            ToolCallRequested(
                calls=[ToolCall(id=f"c{i}", name="x", input={})],
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            )
        )
    reg = ScriptedToolRegistry()
    for _ in range(3):
        reg.push(ToolOutputSuccess(content="ok", truncated=False))
    h = StandardHarness(_config(a, reg, escalation_mode=escalation_mode))
    task = Task.new(
        "do work",
        SessionId("leaf-s1"),
        ReactConfig.per_loop(2),
        budget=BudgetLimits(max_turns=None),
    )
    return await h.run(HarnessRunOptions(task))


async def _run_plan_execute_exhaustion(escalation_mode: object) -> object:
    """A PlanExecute whose EXECUTE child leaf (``PerLoop(2)``) exhausts its own
    cap on the first task and bubbles a ``StrategyOutcomeBudgetExhausted`` up to
    the PlanExecute body, which resolves THIS scope's ``Escalate`` placeholder
    (combinator escalate site 1). A large global ``max_turns`` keeps the global
    backstop from tripping first."""
    a = MockAgent(AgentId("test"))
    # Plan turn: one FinalResponse carrying a single task.
    a.push(FinalResponse(content='{"tasks":["task one"],"rationale":"r"}', usage=_usage()))
    # Execute child: 3 tool-call turns against a PerLoop(2) leaf cap -> exhausts.
    for i in range(3):
        a.push(
            ToolCallRequested(
                calls=[ToolCall(id=f"e{i}", name="x", input={})],
                usage=_usage(),
            )
        )
    reg = ScriptedToolRegistry()
    for _ in range(3):
        reg.push(ToolOutputSuccess(content="ok", truncated=False))
    h = StandardHarness(_config(a, reg, escalation_mode=escalation_mode))
    tree = PlanExecuteConfig(
        plan=ReactConfig(
            budget=ReactConfig.per_loop(2**31 - 1).budget, agent="", toolset="", output=""
        ),
        execute=ReactConfig(budget=ReactConfig.per_loop(2).budget, agent="", toolset=""),
    )
    task = Task.new(
        "build a CLI",
        SessionId("pe-s1"),
        tree,
        budget=BudgetLimits(max_turns=10_000),
    )
    return await h.run(HarnessRunOptions(task))


def _usage(in_t: int = 1, out_t: int = 1) -> TokenUsage:
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


# ===========================================================================
# SurfaceToHuman: a bare-leaf exhaustion PAUSES with a BudgetExhausted request
# (omits Skip); a combinator exhaustion PAUSES (offers Skip).
# ===========================================================================


async def test_surface_to_human_bare_leaf_pauses_omitting_skip() -> None:
    r = await _run_leaf_exhaustion(EscalationModeSurfaceToHuman())
    assert isinstance(r, RunResultWaitingForHuman)
    req = r.request
    assert isinstance(req, HumanRequestBudgetExhausted)
    assert req.phase == "react"
    # Fork C: a bare leaf offers [ContinueWithBudget, Fail] — NO Skip.
    assert [a.kind for a in req.available_actions] == ["continue_with_budget", "fail"]
    # Fork E: the request carries the node's budget counters.
    assert req.steps_taken == 2
    assert req.continues_used == 0
    # The partial is the documented ReAct shape (no FinalResponse this window).
    assert req.partial_output == _react_partial_json("")
    # The preserved state is resumable and carries the same request.
    assert isinstance(r.state, PausedState)
    assert r.state.human_request == req


async def test_surface_to_human_combinator_pauses_offering_skip() -> None:
    r = await _run_plan_execute_exhaustion(EscalationModeSurfaceToHuman())
    assert isinstance(r, RunResultWaitingForHuman)
    req = r.request
    assert isinstance(req, HumanRequestBudgetExhausted)
    assert req.phase == "plan_execute"
    # Fork C: a combinator offers [ContinueWithBudget, Skip, Fail].
    assert [a.kind for a in req.available_actions] == ["continue_with_budget", "skip", "fail"]


# ===========================================================================
# Autonomous: no pause — the existing propagate behavior is unchanged.
# ===========================================================================


async def test_autonomous_does_not_pause_bare_leaf() -> None:
    r = await _run_leaf_exhaustion(EscalationModeAutonomous())
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonBudgetExceeded)
    assert r.reason.limit_type == "turns"


async def test_autonomous_does_not_pause_combinator() -> None:
    r = await _run_plan_execute_exhaustion(EscalationModeAutonomous())
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonBudgetExceeded)


# ===========================================================================
# Resume — Continue / Fail / Skip(leaf) / Skip(PlanExecute).
# ===========================================================================


def _budget_exhausted_state(
    *,
    loop_strategy: object,
    phase: str,
    steps_taken: int,
    available_actions: list[EscalationAction],
) -> PausedState:
    req = HumanRequestBudgetExhausted(
        phase=phase,
        policy=BudgetPolicyTotalSteps(value=steps_taken),
        steps_taken=steps_taken,
        continues_used=0,
        partial_output=_react_partial_json(""),
        available_actions=available_actions,
    )
    task = Task.new(
        "do work",
        SessionId("resume-s1"),
        loop_strategy,  # type: ignore[arg-type]
        budget=BudgetLimits(max_turns=steps_taken),
    )
    return PausedState(
        session_id=SessionId("resume-s1"),
        task_id=task.id,
        turn_number=steps_taken,
        session_state=SessionState(),
        human_request=req,
        task=task,
        budget_used=BudgetSnapshot(turns=steps_taken),
    )


async def test_resume_continue_with_budget_grants_and_reenters() -> None:
    # After granting 3 more steps the restored leaf has room to finish.
    a = MockAgent(AgentId("test"))
    a.push(FinalResponse(content="finished after grant", usage=_usage()))
    h = StandardHarness(_config(a, escalation_mode=EscalationModeSurfaceToHuman()))
    state = _budget_exhausted_state(
        loop_strategy=ReactConfig.per_loop(2),
        phase="react",
        steps_taken=2,
        available_actions=_leaf_escalation_actions(
            BudgetExhausted(
                policy=BudgetPolicyTotalSteps(value=2),
                behavior=BudgetExhaustedEscalate(),
                steps_taken=2,
                continues_used=0,
                phase="react",
            )
        ),
    )
    r = await h.resume(
        state, HumanResponseEscalate(action=EscalationActionContinueWithBudget(steps=3))
    )
    assert isinstance(r, RunResultSuccess)
    assert r.output == "finished after grant"


def test_grant_task_budget_raises_caps() -> None:
    task = Task.new(
        "do work",
        SessionId("g-s1"),
        ReactConfig.per_loop(2),
        budget=BudgetLimits(max_turns=2),
    )
    granted = _grant_task_budget(task, 5)
    assert granted.budget.max_turns == 5
    assert isinstance(granted.loop_strategy, ReactConfig)
    assert granted.loop_strategy.budget.value == 5
    # A cap already above ``granted`` is left untouched.
    not_lowered = _grant_task_budget(granted, 3)
    assert not_lowered.budget.max_turns == 5
    assert not_lowered.loop_strategy.budget.value == 5


async def test_resume_fail_propagates_budget_exceeded_discarding_partial() -> None:
    a = MockAgent(AgentId("test"))
    h = StandardHarness(_config(a, escalation_mode=EscalationModeSurfaceToHuman()))
    state = _budget_exhausted_state(
        loop_strategy=ReactConfig.per_loop(2),
        phase="react",
        steps_taken=2,
        available_actions=[EscalationActionFail()],
    )
    r = await h.resume(state, HumanResponseEscalate(action=EscalationActionFail()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonBudgetExceeded)
    # Fail discards the partial.
    assert r.session_state.messages == []


async def test_resume_skip_bare_leaf_clean_success() -> None:
    a = MockAgent(AgentId("test"))
    h = StandardHarness(_config(a, escalation_mode=EscalationModeSurfaceToHuman()))
    state = _budget_exhausted_state(
        loop_strategy=ReactConfig.per_loop(2),
        phase="react",
        steps_taken=2,
        # Advisory only (fork D): an out-of-set Skip is still honored for a leaf.
        available_actions=[EscalationActionFail()],
    )
    r = await h.resume(state, HumanResponseEscalate(action=EscalationActionSkip()))
    # A leaf has no sibling to skip to -> clean (empty) Success.
    assert isinstance(r, RunResultSuccess)
    assert r.output == ""
    # No agent turn ran.
    assert a.call_count == 0


async def test_resume_skip_plan_execute_advances_outer_loop() -> None:
    a = MockAgent(AgentId("test"))
    # Re-entering the PlanExecute loop from the checkpoint re-plans + runs.
    a.push(FinalResponse(content='{"tasks":["only task"],"rationale":"r"}', usage=_usage()))
    a.push(FinalResponse(content="did only task", usage=_usage()))
    h = StandardHarness(_config(a, escalation_mode=EscalationModeSurfaceToHuman()))
    state = _budget_exhausted_state(
        loop_strategy=PlanExecuteConfig.simple(),
        phase="plan_execute",
        steps_taken=6,
        available_actions=[
            EscalationActionContinueWithBudget(steps=6),
            EscalationActionSkip(),
            EscalationActionFail(),
        ],
    )
    # Give the re-entered run room to finish.
    state.task = _grant_task_budget(state.task, 100)
    state.task.budget.max_turns = 100
    r = await h.resume(state, HumanResponseEscalate(action=EscalationActionSkip()))
    # The PlanExecute outer loop re-drives from the checkpoint (not a leaf
    # short-circuit) — it produces a real terminal.
    assert isinstance(r, RunResultSuccess)


async def test_escalate_response_to_non_budget_pause_halts() -> None:
    # Fork B / out-of-contract: an Escalate response delivered to a non-budget
    # pause halts cleanly rather than mis-resuming.
    a = MockAgent(AgentId("test"))
    h = StandardHarness(_config(a, escalation_mode=EscalationModeSurfaceToHuman()))
    task = Task.new("do work", SessionId("nb-s1"), ReactConfig.per_loop(2))
    state = PausedState(
        session_id=SessionId("nb-s1"),
        task_id=task.id,
        turn_number=1,
        session_state=SessionState(),
        human_request=HumanRequestReview(content="ok?"),
        task=task,
        budget_used=BudgetSnapshot(turns=1),
    )
    r = await h.resume(state, HumanResponseEscalate(action=EscalationActionFail()))
    assert isinstance(r, RunResultFailure)
    assert r.reason.kind == "human_halted"
    assert a.call_count == 0


# ===========================================================================
# Fixture replay — deserialize, assert field-for-field, re-serialize
# byte-identically.
# ===========================================================================


def test_fixture_replay_byte_identical() -> None:
    path = _fixture_path()
    if not path.exists():
        pytest.skip("shared budget_exhausted fixture not present")
    raw = path.read_text()
    parsed = PausedState.model_validate_json(raw)

    # Field-for-field against the canonical value.
    assert parsed.session_id == "sess-130"
    assert parsed.task_id == "task-130"
    assert parsed.turn_number == 6
    req = parsed.human_request
    assert isinstance(req, HumanRequestBudgetExhausted)
    assert req.phase == "plan_execute"
    assert isinstance(req.policy, BudgetPolicyTotalSteps)
    assert req.policy.value == 6
    assert req.steps_taken == 6
    assert req.continues_used == 1
    assert req.partial_output == '{"node":"plan_execute","tasks":2,"ledger":[]}'
    assert [a.kind for a in req.available_actions] == ["continue_with_budget", "skip", "fail"]
    assert isinstance(req.available_actions[0], EscalationActionContinueWithBudget)
    assert req.available_actions[0].steps == 6

    # Re-serializes byte-identically (order-preserving minify).
    canonical = json.dumps(json.loads(raw), separators=(",", ":"))
    assert parsed.model_dump_json() == canonical
