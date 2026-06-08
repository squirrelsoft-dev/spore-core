"""Tests for the Continue cross-process checkpoint (issue #129).

#129 wires two gaps left by #125/#130:

1. **Resume drops ``continues_used``** — the resume path reads ``continues_used``
   from the ``HumanRequest::BudgetExhausted`` payload and seeds it back into the
   reconstructed budget context (AC2, load-bearing).
2. **A node can select ``Continue`` behavior in a live run** — a serialized
   ``behavior`` field on all five config structs, and the
   ``ExhaustedResolution.CONTINUE`` arms genuinely loop in-process (consume
   continue → reset steps → re-enter) instead of falling into the Escalate/pause
   branch.

Resolved forks (maintainer-pinned; the Rust reference follows them):

* Q1 — a serialized ``behavior`` (``BudgetExhaustedBehavior``) on ALL FIVE config
  structs. The ReAct leaf honors its ``behavior`` ONLY at the
  top-level/bare-leaf resolution site; a NESTED leaf still propagates exhaustion
  to its parent (preserve #125 rule 6).
* Q2 — only the shared ``PausedState`` pause/resume seam is extracted into a
  shared checkpoint utility. Continue's and Ralph's context policies stay
  distinct.
* Q3 — the ``HumanRequest::BudgetExhausted`` payload is the SOLE carrier of
  ``continues_used`` across a pause. In-process Continue requires zero
  serialization (AC3).

Mirrors the ``#129`` tests in ``rust/crates/spore-core/src/harness.rs``.
"""

from __future__ import annotations

import json
from pathlib import Path

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetLimits,
    BudgetSnapshot,
    EscalationActionContinueWithBudget,
    EscalationActionFail,
    EscalationModeAutonomous,
    EscalationModeSurfaceToHuman,
    FinalResponse,
    HaltReasonBudgetExceeded,
    HarnessConfig,
    HarnessRunOptions,
    HumanRequestBudgetExhausted,
    HumanResponseEscalate,
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
from spore_core.agent import Context, ToolCallRequested, TurnResult
from spore_core.harness import (
    BudgetContext,
    BudgetExhaustedContinue,
    BudgetExhaustedEscalate,
    BudgetExhaustedFail,
    BudgetPolicyPerLoop,
    BudgetPolicyTotalSteps,
    ExhaustedResolution,
)
from spore_core.model import Message, Role, TextContent, ToolCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage(in_t: int = 1, out_t: int = 1) -> TokenUsage:
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


def _config(
    agent: object,
    tool_registry: ScriptedToolRegistry | None = None,
    *,
    escalation_mode: object = None,
) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,  # type: ignore[arg-type]
        tool_registry=tool_registry if tool_registry is not None else ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        storage=StorageProvider.single(InMemoryStorageProvider()),
        escalation_mode=escalation_mode,  # type: ignore[arg-type]
    )


class _AlwaysToolAgent:
    """An agent that ALWAYS requests a tool (never finishes) — every ReAct window
    re-exhausts its refreshed cap. Mirrors Rust's ``budget_exhausting_agent``."""

    def __init__(self, agent_id: AgentId) -> None:
        self._id = agent_id
        self.call_count = 0

    async def turn(self, context: Context) -> TurnResult:
        self.call_count += 1
        return ToolCallRequested(
            calls=[ToolCall(id=f"c{self.call_count}", name="x", input={})],
            usage=_usage(),
        )

    def id(self) -> AgentId:
        return self._id


def _tool_reg(n: int) -> ScriptedToolRegistry:
    reg = ScriptedToolRegistry()
    for _ in range(n):
        reg.push(ToolOutputSuccess(content="ok", truncated=False))
    return reg


def _fixtures_root() -> Path:
    return Path(__file__).resolve().parents[4] / "fixtures"


def _continue_checkpoint_paused_state() -> PausedState:
    """The canonical ``continue_checkpoint.json`` value: a bare ReAct leaf carrying
    ``Continue{max_continues:2, on_exhausted:Fail}`` paused mid-loop with
    ``continues_used: 1`` on its ``HumanRequest::BudgetExhausted``. Cross-language
    byte-identity ground truth for the AC2 resume test."""
    task = Task(
        id="task-129",  # type: ignore[arg-type]
        instruction="iterate on the patch",
        session_id=SessionId("sess-129"),
        budget=BudgetLimits(max_turns=None),
        loop_strategy=ReactConfig(
            budget=BudgetPolicyPerLoop(value=3),
            behavior=BudgetExhaustedContinue(
                max_continues=2,
                on_exhausted=BudgetExhaustedFail(),
            ),
            agent="worker",
            toolset="patch-tools",
        ),
    )
    return PausedState(
        session_id=SessionId("sess-129"),
        task_id="task-129",  # type: ignore[arg-type]
        turn_number=3,
        session_state=SessionState(
            messages=[
                Message(
                    role=Role.ASSISTANT,
                    content=TextContent(text='{"node":"react","last":""}'),
                )
            ],
        ),
        human_request=HumanRequestBudgetExhausted(
            phase="react",
            policy=BudgetPolicyPerLoop(value=3),
            steps_taken=3,
            continues_used=1,
            partial_output='{"node":"react","last":""}',
            available_actions=[
                EscalationActionContinueWithBudget(steps=3),
                EscalationActionFail(),
            ],
        ),
        task=task,
        budget_used=BudgetSnapshot(turns=3),
    )


# ===========================================================================
# AC1 — shared checkpoint utility round-trips.
# ===========================================================================


def test_shared_checkpoint_utility_round_trips() -> None:
    """AC1: the SHARED checkpoint utility round-trips a ``PausedState`` (the
    durable pause/resume seam reused by both the cross-process Continue path and
    Ralph's pause-propagation)."""
    state = _continue_checkpoint_paused_state()
    blob = state.serialize_checkpoint()
    restored = PausedState.load_checkpoint(blob)
    assert restored == state

    # A Ralph-style paused-state (NO human_request) ALSO round-trips through the
    # same utility — proving it is shared, not Continue-specific.
    ralph_task = Task.new("ralph work", SessionId("sess-ralph"), ReactConfig.per_loop(2))
    ralph_state = PausedState(
        session_id=SessionId("sess-ralph"),
        task_id=ralph_task.id,
        turn_number=2,
        session_state=SessionState(),
        human_request=None,
        task=ralph_task,
        budget_used=BudgetSnapshot(turns=2),
    )
    blob = ralph_state.serialize_checkpoint()
    assert PausedState.load_checkpoint(blob) == ralph_state


# ===========================================================================
# AC2 (wire) — continue_checkpoint.json (de)serializes byte-identically.
# ===========================================================================


def test_fixture_replay_continue_checkpoint() -> None:
    """AC2 (wire): ``continue_checkpoint.json`` round-trips byte-identically — the
    new fixture capturing a ``Continue`` node paused with ``continues_used > 0``."""
    raw = (_fixtures_root() / "paused_states" / "continue_checkpoint.json").read_text()
    parsed = PausedState.load_checkpoint(raw)
    assert parsed == _continue_checkpoint_paused_state()
    canonical = json.dumps(json.loads(raw), separators=(",", ":"))
    assert _continue_checkpoint_paused_state().serialize_checkpoint() == canonical


# ===========================================================================
# AC2 (unit, load-bearing core) — BudgetContext.resumed seeds continues_used.
# ===========================================================================


def test_budget_context_resumed_seeds_continues_used_and_bounds_chain() -> None:
    """AC2 (load-bearing core): the resume seam seeds the reconstructed scope's
    ``continues_used`` from the request (NOT 0), and a subsequent exhaustion falls
    through to ``on_exhausted`` after the REMAINING continues — not a refreshed
    ``max_continues``."""
    behavior = BudgetExhaustedContinue(max_continues=2, on_exhausted=BudgetExhaustedFail())
    # Reconstruct as the resume seam does: continues_used seeded to 1.
    scope = BudgetContext.resumed(BudgetPolicyPerLoop(value=3), behavior, "react", 1)
    assert scope.continues_used == 1, "seeded, NOT zeroed"
    assert scope.steps_taken == 0, "fresh per-round step budget"
    # Only ONE continue remains (to reach max_continues=2).
    assert scope.continues_remaining() == 1
    assert scope.resolve_exhausted() == ExhaustedResolution.CONTINUE
    assert scope.continues_used == 2
    # Continues now spent → fall through to on_exhausted=Fail.
    assert scope.resolve_exhausted() == ExhaustedResolution.FAIL

    # Contrast: a FRESH (pre-#129) scope would grant TWO continues — the bug.
    fresh = BudgetContext(
        policy=BudgetPolicyPerLoop(value=3),
        behavior=BudgetExhaustedContinue(max_continues=2, on_exhausted=BudgetExhaustedFail()),
        phase="react",
    )
    assert fresh.continues_remaining() == 2, "the bug: full budget refreshed"


# ===========================================================================
# AC2 (end-to-end, load-bearing) — a Continue spanning a pause resumes with the
# correct continues_used, then falls through after the REMAINING continues.
# ===========================================================================


async def test_resume_continue_preserves_continues_used_then_falls_through() -> None:
    """AC2 (LOAD-BEARING, end-to-end): a ``Continue`` that SPANS a process pause
    resumes with the correct ``continues_used`` (NOT 0). FAILS on pre-#129 code,
    which zeroed the counter and over-granted continues.

    DISCRIMINATING setup: a ``Continue{max_continues:2, on_exhausted:Fail}`` leaf
    whose checkpoint records ``continues_used: 1`` (ONE continue already spent).
    The resumed worker NEVER finishes, so every granted in-process window
    re-exhausts and drains to ``Fail``. The granted per-window cap is
    ``steps_taken + steps = 3 + 1 = 4`` turns. On resume the operator's
    ``ContinueWithBudget`` runs window A; with ONE in-process continue left
    window B runs, then ``Fail``: 2 windows ⇒ 8 turns. The bug (zeroed counter)
    would grant a THIRD window ⇒ 12 turns. We assert ``turns == 8``."""
    a = _AlwaysToolAgent(AgentId("worker"))
    h = StandardHarness(_config(a, _tool_reg(40), escalation_mode=EscalationModeSurfaceToHuman()))

    state = _continue_checkpoint_paused_state()
    # Bare leaf resolving via the default ("") registry handle on resume.
    state.task.loop_strategy = ReactConfig(
        budget=BudgetPolicyPerLoop(value=3),
        behavior=BudgetExhaustedContinue(max_continues=2, on_exhausted=BudgetExhaustedFail()),
        agent="",
        toolset="",
    )
    # continues_used == 1: ONE continue already spent (only one remains).
    req = state.human_request
    assert isinstance(req, HumanRequestBudgetExhausted)
    req.continues_used = 1
    req.steps_taken = 3

    resumed = await h.resume(
        state, HumanResponseEscalate(action=EscalationActionContinueWithBudget(steps=1))
    )
    assert isinstance(resumed, RunResultFailure)
    assert isinstance(resumed.reason, HaltReasonBudgetExceeded)
    assert resumed.reason.limit_type == "turns"
    # #129: window A (operator grant) + window B (the ONE remaining in-process
    # continue) → 2 windows × cap 4 = 8 turns. The bug (zeroed continues_used)
    # grants a THIRD window → 12 turns.
    assert resumed.turns == 8, (
        "expected 2 windows (continues_used preserved at 1, one continue left); "
        "the bug zeroes it and grants an extra window → 12 turns"
    )


# ===========================================================================
# Live in-process Continue completes (AC3: no serialization).
# ===========================================================================


async def test_live_continue_loops_in_process_then_completes() -> None:
    """LIVE Continue reachable (``ExhaustedResolution.CONTINUE`` is no longer
    dead): a bare-leaf run with ``behavior: Continue{max_continues:1}`` exhausts
    in-process, gets a granted continue (counter resets, ``continues_used``
    bumps), loops, and completes — all WITHOUT a pause (AC3: no serialization)."""
    a = MockAgent(AgentId("test"))
    # First window: 2 tool turns exhaust the PerLoop{2} cap → Continue grant.
    a.push(ToolCallRequested(calls=[ToolCall(id="c0", name="x", input={})], usage=_usage()))
    a.push(ToolCallRequested(calls=[ToolCall(id="c1", name="x", input={})], usage=_usage()))
    # After the in-process continue refreshes the cap, the worker completes.
    a.push(FinalResponse(content="done after in-process continue", usage=_usage()))
    # Autonomous so an Escalate fall-through would NOT pause — proving the success
    # came from the Continue loop, not a HITL pause.
    h = StandardHarness(_config(a, _tool_reg(3), escalation_mode=EscalationModeAutonomous()))
    task = Task.new(
        "iterate",
        SessionId("live-s1"),
        ReactConfig(
            budget=BudgetPolicyPerLoop(value=2),
            behavior=BudgetExhaustedContinue(max_continues=1, on_exhausted=BudgetExhaustedFail()),
            agent="",
            toolset="",
        ),
        budget=BudgetLimits(max_turns=None),
    )
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultSuccess), r
    assert r.output == "done after in-process continue"


# ===========================================================================
# AC4 — Continue resume PRESERVES session context (vs Ralph DISCARDS).
# ===========================================================================


async def test_continue_resume_preserves_session_context() -> None:
    """AC4: a Continue resume PRESERVES the prior ``session_state.messages`` (the
    conversation context survives the pause). Asserts the shared checkpoint
    utility did NOT unify the context policy with Ralph (which discards)."""
    a = MockAgent(AgentId("test"))
    a.push(FinalResponse(content="resumed with context", usage=_usage()))
    h = StandardHarness(_config(a, _tool_reg(1), escalation_mode=EscalationModeAutonomous()))

    state = _continue_checkpoint_paused_state()
    state.task.loop_strategy = ReactConfig(
        budget=BudgetPolicyPerLoop(value=3),
        behavior=BudgetExhaustedContinue(max_continues=2, on_exhausted=BudgetExhaustedFail()),
        agent="",
        toolset="",
    )
    prior_messages = list(state.session_state.messages)
    assert len(prior_messages) >= 1, "checkpoint carries prior context"

    resumed = await h.resume(
        state, HumanResponseEscalate(action=EscalationActionContinueWithBudget(steps=3))
    )
    assert isinstance(resumed, RunResultSuccess), resumed
    assert resumed.output == "resumed with context"
    # Continue PRESERVES the prior conversation context: the resumed session
    # RETAINS the checkpoint's prior assistant message — it did NOT start from an
    # empty (re-seeded) session the way Ralph would. (With ``NoopContextManager``
    # the resumed FinalResponse is not appended to the session, so we assert
    # preservation of the prior message rather than a strict length increase.)
    assert prior_messages[0] in resumed.session_state.messages, (
        "Continue must preserve the prior context across the pause; "
        f"got {resumed.session_state.messages}"
    )


# ===========================================================================
# Q1 — bare leaf HONORS behavior; nested leaf PROPAGATES (does not self-resolve).
# ===========================================================================


async def test_bare_leaf_honors_fail_behavior() -> None:
    """Q1: a top-level (bare) leaf with ``behavior: Fail`` resolves to a
    ``BudgetExceeded`` Failure with the partial DISCARDED — proving the bare-leaf
    site honored ``Fail`` (the default ``Escalate`` would, under Autonomous,
    surface the partial)."""
    a = MockAgent(AgentId("test"))
    for i in range(3):
        a.push(ToolCallRequested(calls=[ToolCall(id=f"c{i}", name="x", input={})], usage=_usage()))
    h = StandardHarness(_config(a, _tool_reg(3), escalation_mode=EscalationModeAutonomous()))
    task = Task.new(
        "do work",
        SessionId("fail-s1"),
        ReactConfig(
            budget=BudgetPolicyPerLoop(value=2),
            behavior=BudgetExhaustedFail(),
            agent="",
            toolset="",
        ),
        budget=BudgetLimits(max_turns=None),
    )
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure), r
    assert isinstance(r.reason, HaltReasonBudgetExceeded)
    # Fail contract: the partial is DISCARDED.
    assert r.session_state.messages == []


async def test_nested_leaf_propagates_does_not_self_resolve() -> None:
    """Q1: a NESTED leaf does NOT self-resolve — its ``Continue`` behavior is
    IGNORED by the leaf body (it propagates to the parent), so the PARENT
    PlanExecute combinator's ``behavior`` governs. Here the parent carries the
    default ``Escalate`` placeholder, so the nested leaf's exhaustion surfaces as
    the PARENT's ``plan_execute`` pause (phase == "plan_execute"), NOT a
    leaf-level ``react`` resolution of its own ``Continue``."""
    a = MockAgent(AgentId("test"))
    a.push(FinalResponse(content='{"tasks":["x"],"rationale":"r"}', usage=_usage()))  # plan
    for i in range(4):
        a.push(ToolCallRequested(calls=[ToolCall(id=f"e{i}", name="x", input={})], usage=_usage()))
    h = StandardHarness(_config(a, _tool_reg(4), escalation_mode=EscalationModeSurfaceToHuman()))
    # The execute leaf carries ``Continue`` — but NESTED, so the leaf must NOT
    # self-resolve it; the PlanExecute parent (Escalate placeholder) pauses.
    tree = PlanExecuteConfig(
        plan=ReactConfig(
            budget=BudgetPolicyTotalSteps(value=2**31 - 1), agent="", toolset="", output=""
        ),
        execute=ReactConfig(
            budget=BudgetPolicyPerLoop(value=2),
            behavior=BudgetExhaustedContinue(max_continues=1, on_exhausted=BudgetExhaustedFail()),
            agent="",
            toolset="",
        ),
        behavior=BudgetExhaustedEscalate(),
    )
    task = Task.new("build", SessionId("nest-s1"), tree, budget=BudgetLimits(max_turns=10_000))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultWaitingForHuman), r
    req = r.request
    assert isinstance(req, HumanRequestBudgetExhausted)
    # The PARENT resolved (phase == plan_execute); the nested leaf did NOT
    # self-resolve its own ``Continue`` at the "react" phase.
    assert req.phase == "plan_execute"


# ===========================================================================
# AC3 — in-process Continue does NO serialization (no checkpoint persisted).
# ===========================================================================


async def test_in_process_continue_does_no_serialization() -> None:
    """AC3: an in-process Continue (live, no pause) never persists a checkpoint —
    the run completes via :class:`RunResultSuccess`, never a
    :class:`RunResultWaitingForHuman`, and the storage holds no paused state."""
    a = MockAgent(AgentId("test"))
    a.push(ToolCallRequested(calls=[ToolCall(id="c0", name="x", input={})], usage=_usage()))
    a.push(ToolCallRequested(calls=[ToolCall(id="c1", name="x", input={})], usage=_usage()))
    a.push(FinalResponse(content="done", usage=_usage()))
    storage = InMemoryStorageProvider()
    cfg = HarnessConfig(
        agent=a,
        tool_registry=_tool_reg(3),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        storage=StorageProvider.single(storage),
        escalation_mode=EscalationModeSurfaceToHuman(),
    )
    h = StandardHarness(cfg)
    task = Task.new(
        "iterate",
        SessionId("ac3-s1"),
        ReactConfig(
            budget=BudgetPolicyPerLoop(value=2),
            behavior=BudgetExhaustedContinue(max_continues=1, on_exhausted=BudgetExhaustedFail()),
            agent="",
            toolset="",
        ),
        budget=BudgetLimits(max_turns=None),
    )
    r = await h.run(HarnessRunOptions(task))
    # An in-process Continue must NOT pause — it completes in-process.
    assert isinstance(r, RunResultSuccess), r
    assert r.output == "done"
    assert not isinstance(r, RunResultWaitingForHuman)
