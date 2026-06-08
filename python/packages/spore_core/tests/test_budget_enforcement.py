"""Per-node budget enforcement + failure isolation tests (issue #125).

#123 built ``BudgetContext`` / ``BudgetStack`` / ``charge`` as pure-arithmetic
scaffold; #124 wired enforcement through the LEGACY ``task.budget.max_turns``
path. #125 makes ``charge`` the REAL per-node enforcement point and
``StrategyOutcomeBudgetExhausted`` a real, isolated, parent-inspectable value.

Covers every rule from the slice plan:
  1. A node capped at N stops at N WITHOUT killing siblings.
  2. In-process Continue resets the counter, honors max_continues, then falls
     through; session/messages unchanged across resets.
  3. Fail yields partial_output = None.
  4. A child BudgetExhausted reaches the parent as a StrategyOutcome, never
     auto-propagated (parent's own scope unaffected; parent can then Complete).
  5. partial_output concrete per node (4 shapes).
  6. The ReAct leaf does not carry its own behavior; it propagates to parent.
  7. Never auto-cascade a child exhaustion into a parent exhaustion.

No new fixtures (fork #3): every type here is runtime-only. Mirrors the Rust
``budget_enforcement_tests`` mod in ``rust/crates/spore-core/src/harness.rs``.
"""

from __future__ import annotations

import json

import pytest

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetLimits,
    ExecutionContext,
    ExecutionRegistry,
    HaltReasonBudgetExceeded,
    HarnessConfig,
    HarnessRunOptions,
    InMemoryStorageProvider,
    MockAgent,
    NoopContextManager,
    ReactConfig,
    RunResultFailure,
    ScriptedToolRegistry,
    SessionId,
    StandardHarness,
    StorageProvider,
    Task,
    TokenUsage,
)
from spore_core.agent import ToolCallRequested
from spore_core.harness import (
    BudgetContext,
    BudgetExhausted,
    BudgetExhaustedContinue,
    BudgetExhaustedEscalate,
    BudgetExhaustedFail,
    BudgetPolicyPerLoop,
    BudgetPolicyTotalSteps,
    BudgetStack,
    ExhaustedResolution,
    StrategyOutcomeBudgetExhausted,
    _hill_climbing_partial_json,
    _plan_execute_partial_json,
    _promote_budget_exhausted,
    _react_partial_json,
    _self_verifying_partial_json,
)
from spore_core import ToolOutputSuccess
from spore_core.model import Role, TextContent, ToolCall
from spore_core.tasklist import TaskList, TaskStatus


def _continue_then_fail(max_continues: int) -> BudgetExhaustedContinue:
    return BudgetExhaustedContinue(max_continues=max_continues, on_exhausted=BudgetExhaustedFail())


# ---------------------------------------------------------------------------
# resolve_exhausted: Fail -> Fail; Escalate -> Escalate
# ---------------------------------------------------------------------------


def test_resolve_fail_and_escalate_terminal() -> None:
    fail = BudgetContext(
        policy=BudgetPolicyPerLoop(value=1), behavior=BudgetExhaustedFail(), phase="p"
    )
    assert fail.resolve_exhausted() == ExhaustedResolution.FAIL

    esc = BudgetContext(
        policy=BudgetPolicyPerLoop(value=1), behavior=BudgetExhaustedEscalate(), phase="p"
    )
    assert esc.resolve_exhausted() == ExhaustedResolution.ESCALATE


# ---------------------------------------------------------------------------
# Rule 2: Continue resets counter, honors max_continues, falls through
# ---------------------------------------------------------------------------


def test_continue_resets_counter_then_falls_through_to_fail() -> None:
    cx = BudgetContext(
        policy=BudgetPolicyTotalSteps(value=3),
        behavior=_continue_then_fail(2),
        phase="phase",
    )

    # 1st exhaustion -> Continue (counter resets to 0, continues_used = 1).
    cx.charge(3)
    with pytest.raises(BudgetExhausted):
        cx.charge(1)
    assert cx.resolve_exhausted() == ExhaustedResolution.CONTINUE
    assert cx.steps_taken == 0
    assert cx.continues_used == 1

    # 2nd exhaustion -> Continue (counter resets again, continues_used = 2).
    cx.charge(3)
    with pytest.raises(BudgetExhausted):
        cx.charge(1)
    assert cx.resolve_exhausted() == ExhaustedResolution.CONTINUE
    assert cx.steps_taken == 0
    assert cx.continues_used == 2

    # 3rd exhaustion -> continues spent -> fall through to Fail.
    cx.charge(3)
    with pytest.raises(BudgetExhausted):
        cx.charge(1)
    assert cx.resolve_exhausted() == ExhaustedResolution.FAIL
    # continues_used does NOT advance past max_continues on the fall-through.
    assert cx.continues_used == 2


def test_continue_in_process_only_session_messages_untouched() -> None:
    """The in-process Continue is a pure counter reset — there is no session /
    messages object on a BudgetContext, and consume_continue only rewinds the
    step counter and bumps continues_used (no serialization, fork #3)."""
    cx = BudgetContext(
        policy=BudgetPolicyPerLoop(value=2),
        behavior=_continue_then_fail(3),
        phase="c",
    )
    cx.charge(2)
    assert cx.steps_taken == 2
    cx.consume_continue()
    assert cx.steps_taken == 0
    assert cx.continues_used == 1
    # The allowance is intact — a fresh round can charge again.
    cx.charge(2)
    assert cx.steps_taken == 2


def test_continue_chain_shares_counter_then_escalates() -> None:
    """The chain shares ONE continues_used counter. Outer Continue{2} grants 2
    continues; once spent the shared counter (2) >= the nested Continue{2}'s max,
    so the nested layer grants nothing and falls straight through to Escalate."""
    behavior = BudgetExhaustedContinue(
        max_continues=2,
        on_exhausted=BudgetExhaustedContinue(
            max_continues=2, on_exhausted=BudgetExhaustedEscalate()
        ),
    )
    cx = BudgetContext(policy=BudgetPolicyPerLoop(value=1), behavior=behavior, phase="chain")

    assert cx.resolve_exhausted() == ExhaustedResolution.CONTINUE
    assert cx.continues_used == 1
    assert cx.resolve_exhausted() == ExhaustedResolution.CONTINUE
    assert cx.continues_used == 2
    # Outer spent -> fall through to nested Continue{2}; the SHARED counter is
    # already 2 so the nested layer grants nothing -> Escalate.
    assert cx.resolve_exhausted() == ExhaustedResolution.ESCALATE


# ---------------------------------------------------------------------------
# Rule 3: Fail -> partial_output = None; Escalate -> partial present
# ---------------------------------------------------------------------------


def test_promote_fail_drops_partial_escalate_keeps_it() -> None:
    err = BudgetExhausted(
        policy=BudgetPolicyPerLoop(value=2),
        behavior=BudgetExhaustedFail(),
        steps_taken=2,
        continues_used=0,
        phase="react",
    )
    failed = _promote_budget_exhausted(err, None)
    assert isinstance(failed, StrategyOutcomeBudgetExhausted)
    assert failed.partial_output is None
    assert failed.steps_taken == 2

    escalated = _promote_budget_exhausted(err, _react_partial_json("the answer so far"))
    assert isinstance(escalated, StrategyOutcomeBudgetExhausted)
    assert escalated.partial_output is not None
    assert "the answer so far" in escalated.partial_output


# ---------------------------------------------------------------------------
# Rule 5: each node's partial_output has its documented shape
# ---------------------------------------------------------------------------


def test_react_partial_shape() -> None:
    data = json.loads(_react_partial_json("hello world"))
    assert data["node"] == "react"
    assert data["last_final_response"] == "hello world"


def test_plan_execute_partial_shape() -> None:
    tl = TaskList()
    a = tl.add("task a")
    tl.add("task b")
    tl.update(a, TaskStatus.IN_PROGRESS)
    tl.complete(a)
    data = json.loads(_plan_execute_partial_json(tl))
    assert data["node"] == "plan_execute"
    assert data["tasks"] == 2
    ledger = data["ledger"]
    assert len(ledger) == 2
    assert ledger[0]["description"] == "task a"
    assert ledger[0]["status"] == "completed"
    assert ledger[1]["status"] == "pending"


def test_self_verifying_partial_shape() -> None:
    data = json.loads(_self_verifying_partial_json("worker output", "verdict: not yet"))
    assert data["node"] == "self_verifying"
    assert data["last_worker_result"] == "worker output"
    assert data["last_verdict"] == "verdict: not yet"


def test_hill_climbing_partial_shape() -> None:
    data = json.loads(_hill_climbing_partial_json(0.875))
    assert data["node"] == "hill_climbing"
    assert data["best_candidate"] == 0.875
    assert data["score"] == 0.875


# ---------------------------------------------------------------------------
# Rule 1: a node capped at N stops at N without killing siblings
# ---------------------------------------------------------------------------


def test_sibling_isolation_fresh_context_per_node() -> None:
    budgets = BudgetStack()
    # Sibling A: capped at 2, exhausts.
    budgets.push(
        BudgetContext(
            policy=BudgetPolicyPerLoop(value=2),
            behavior=BudgetExhaustedEscalate(),
            phase="child-a",
        )
    )
    budgets.current().charge(2)  # type: ignore[union-attr]
    with pytest.raises(BudgetExhausted):
        budgets.current().charge(1)  # type: ignore[union-attr]
    a = budgets.pop()
    assert a is not None
    assert a.steps_taken == 2

    # Sibling B gets a FRESH BudgetContext — A's exhaustion did not bleed in.
    budgets.push(
        BudgetContext(
            policy=BudgetPolicyPerLoop(value=4),
            behavior=BudgetExhaustedEscalate(),
            phase="child-b",
        )
    )
    b = budgets.current()
    assert b is not None
    assert b.steps_taken == 0
    b.charge(3)  # B runs with its own untouched allowance
    assert b.steps_taken == 3


# ---------------------------------------------------------------------------
# Rule 4 & 7: a child exhaustion does NOT auto-cascade to the parent
# ---------------------------------------------------------------------------


def test_child_exhaustion_does_not_charge_parent_scope() -> None:
    budgets = BudgetStack()
    # Parent scope (TotalSteps{5}) that is ALREADY nearly exhausted: it has spent
    # 4 of its 5 steps, leaving EXACTLY ONE remaining. This is the adversarial
    # case for rule 4/7 — if a child's exhaustion auto-cascaded even a single step
    # onto the parent, the parent would be pushed over its own cap and could no
    # longer Complete.
    budgets.push(
        BudgetContext(
            policy=BudgetPolicyTotalSteps(value=5),
            behavior=BudgetExhaustedFail(),
            phase="parent",
        )
    )
    budgets.current().charge(4)  # type: ignore[union-attr]
    assert budgets.current().remaining() == 1, (  # type: ignore[union-attr]
        "parent starts with exactly 1 step left"
    )

    # Child descends with its OWN scope (capped at 1) and exhausts.
    budgets.push(
        BudgetContext(
            policy=BudgetPolicyPerLoop(value=1),
            behavior=BudgetExhaustedEscalate(),
            phase="child",
        )
    )
    budgets.current().charge(1)  # type: ignore[union-attr]
    with pytest.raises(BudgetExhausted):
        budgets.current().charge(1)  # type: ignore[union-attr]
    # The child surfaces a BudgetExhausted value — modelled here by popping its
    # scope. Crucially the parent scope is UNCHARGED by this.
    budgets.pop()

    parent = budgets.current()
    assert parent is not None
    assert parent.steps_taken == 4, (
        "rule 4/7: child exhaustion did NOT auto-charge the parent — its "
        "steps_taken is unchanged at 4 (not bumped to 5)"
    )
    assert parent.remaining() == 1, (
        "the parent STILL has its 1 remaining step after the child exhausted"
    )
    # The parent can spend its last step and Complete — proving the child's
    # exhaustion did NOT push it over its own cap.
    parent.charge(1)
    assert parent.steps_taken == 5
    # And only now is the parent itself exhausted (at its own 5, by its own work —
    # never by the child's).
    with pytest.raises(BudgetExhausted):
        parent.charge(1)


# ---------------------------------------------------------------------------
# ExecutionContext helpers: push / charge / resolve / pop round-trip
# ---------------------------------------------------------------------------


def test_execution_context_charge_and_resolve_round_trip() -> None:
    cx = ExecutionContext(registry=ExecutionRegistry.empty())
    assert cx.budgets.depth() == 0

    cx.push_budget(BudgetPolicyPerLoop(value=2), _continue_then_fail(1), "node")
    assert cx.budgets.depth() == 1
    assert cx.charge_current(2) is None
    assert cx.charge_current(1) is not None  # scope exhausts at its cap
    assert cx.resolve_current() == ExhaustedResolution.CONTINUE
    # After the continue, the counter reset -> charging is possible again.
    assert cx.charge_current(2) is None
    assert cx.charge_current(1) is not None
    assert cx.resolve_current() == ExhaustedResolution.FAIL
    popped = cx.pop_budget()
    assert popped is not None
    assert cx.budgets.depth() == 0


def test_charge_with_no_scope_never_exhausts() -> None:
    cx = ExecutionContext(registry=ExecutionRegistry.empty())
    assert cx.charge_current(2**31 - 1) is None
    assert cx.resolve_current() == ExhaustedResolution.FAIL


# ---------------------------------------------------------------------------
# Rule 6 (integration): when the ReAct LEAF's OWN policy is the binding cap,
# the leaf PROPAGATES a typed BudgetExhausted that surfaces as a BudgetExceeded
# terminal carrying the partial — it never self-resolves Continue/Fail.
# ---------------------------------------------------------------------------


def _config(agent: MockAgent, tool_registry: ScriptedToolRegistry) -> HarnessConfig:
    # #130: this suite asserts the #125 ``Escalate`` PROPAGATE behavior (a budget
    # exhaustion surfaces as a ``BudgetExceeded`` terminal). Under the default
    # ``SurfaceToHuman`` mode that exhaustion now PAUSES, so pin ``Autonomous``
    # explicitly — mirroring the Rust ``standard_config`` used by the propagate
    # tests.
    from spore_core import EscalationModeAutonomous

    return HarnessConfig(
        agent=agent,
        tool_registry=tool_registry,
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        storage=StorageProvider.single(InMemoryStorageProvider()),
        escalation_mode=EscalationModeAutonomous(),
    )


async def test_react_leaf_cap_binding_propagates_partial() -> None:
    a = MockAgent(AgentId("test"))
    # The leaf cap is 2; push 3 tool-call turns so the window runs 2 turns and
    # hits the leaf's own cap (no global max_turns set).
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
    h = StandardHarness(_config(a, reg))
    # Leaf PerLoop(2), NO global cap -> the leaf policy is the binding cap.
    task = Task.new(
        "do work",
        SessionId("leaf-s1"),
        ReactConfig.per_loop(2),
        budget=BudgetLimits(max_turns=None),
    )
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonBudgetExceeded)
    assert r.reason.limit_type == "turns"
    # The turn count is the EXHAUSTED NODE's own ``steps_taken`` (#125
    # BudgetExhausted path), which equals the leaf cap N=2 here.
    assert r.turns == 2
    # The #125 discriminator: the partial flowed through
    # ``StrategyOutcomeBudgetExhausted.partial_output`` -> ``_drive_strategy``'s
    # BudgetExhausted arm, which materializes the partial as a SINGLE assistant
    # Text message. On PRE-#125 code the leaf's recorded terminal surfaced the
    # window's RAW session (the tool-call / observation messages), NOT this
    # node-concrete partial JSON — so these assertions FAIL on the old path and
    # PROVE the new BudgetExhausted machinery is exercised.
    assert len(r.session_state.messages) == 1, (
        "BudgetExhausted arm materializes exactly the partial as one assistant "
        "message (not the window's raw transcript)"
    )
    msg = r.session_state.messages[0]
    assert msg.role == Role.ASSISTANT
    assert isinstance(msg.content, TextContent)
    text = msg.content.text
    # The documented ReAct partial shape (fork #2): the last FinalResponse text as
    # JSON. This window produced no FinalResponse, so the documented shape carries
    # an empty ``last_final_response``.
    assert text == _react_partial_json("")
    # Sanity: the materialized text is genuinely the partial helper's output, i.e.
    # valid JSON with the node tag — not free-form prose.
    parsed = json.loads(text)
    assert parsed["node"] == "react"
    assert parsed["last_final_response"] == ""
