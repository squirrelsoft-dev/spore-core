"""Regression tests for two cordyceps-composition harness fixes (#131), ported
from the Rust reference (``rust/crates/spore-core/src/harness.rs``).

Fix A — Ralph FILLS the worker leaf agent, never REPLACES it. The function
formerly ``_override_worker_agent`` is now ``_fill_empty_worker_agent``: it
supplies Ralph's ``agent`` to a worker leaf ONLY where the leaf left its own
handle empty. An explicitly-declared leaf agent (the architect's node) is
authoritative and is never overwritten.

Fix B — the plan phase runs under the plan sub-strategy's OWN declared budget
(e.g. a ReAct ``PerLoop{4}``); the global ``max_turns`` is only the outer
backstop. The previous ``min(global, turns + 1)`` clamp pinned the planner to a
SINGLE turn, starving multi-step task-graph authoring (the planner could not
both call a tool and finish, so it never emitted the plan JSON).

Each test is constructed to FAIL on the pre-fix behavior.
"""

from __future__ import annotations

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetLimits,
    BudgetPolicyPerLoop,
    FinalResponse,
    HarnessConfig,
    HarnessRunOptions,
    InMemoryStorageProvider,
    MockAgent,
    PlanExecuteConfig,
    RalphConfig,
    ReactConfig,
    RunResultSuccess,
    ScriptedToolRegistry,
    SelfVerifyingConfig,
    SessionId,
    SessionState,
    StandardHarness,
    StorageProvider,
    Task,
    TokenUsage,
    ToolCall,
    ToolCallRequested,
    ToolOutputSuccess,
)
from spore_core.harness import (
    _fill_empty_worker_agent,
    _worker_agent_key_of,
)


# ---------------------------------------------------------------------------
# Shared cordyceps shape: Ralph[ PlanExecute[ ReAct{planner},
#                                             SelfVerifying[ ReAct{executor} ] ] ]
# with a NON-EMPTY Ralph agent. The executor leaf is EXPLICITLY declared.
# ---------------------------------------------------------------------------


def _cordyceps_pe(*, executor_agent: str) -> PlanExecuteConfig:
    return PlanExecuteConfig(
        plan=ReactConfig(
            budget=BudgetPolicyPerLoop(value=4),
            agent="planner",
            toolset="",
            output="",
        ),
        execute=SelfVerifyingConfig(
            inner=ReactConfig(
                budget=BudgetPolicyPerLoop(value=12),
                agent=executor_agent,
                toolset="",
                output="",
            ),
            evaluator="",
        ),
    )


# ---------------------------------------------------------------------------
# Fix A: a non-empty Ralph agent FILLS only the empty leaf; an explicitly
# declared leaf agent is authoritative and is NEVER shadowed.
#
# Pre-fix (``_override_worker_agent``) unconditionally rewrote the worker leaf,
# so the explicit ``executor`` handle below would become ``ralph-agent`` — this
# test asserts it stays ``executor``.
# ---------------------------------------------------------------------------


def test_fill_does_not_shadow_explicit_worker_agent() -> None:
    pe = _cordyceps_pe(executor_agent="executor")
    filled = _fill_empty_worker_agent(pe, "ralph-agent")

    # The explicitly-declared worker leaf wins — Ralph's agent does NOT replace it.
    assert _worker_agent_key_of(filled.execute.inner) == "executor", (
        "the architect's explicit executor leaf must not be shadowed by Ralph"
    )
    # The original tree is untouched (a copy was returned).
    assert _worker_agent_key_of(pe.execute.inner) == "executor"


# An EMPTY worker leaf is still FILLED with Ralph's agent (the bare-leaf
# convenience the fill function preserves).
def test_fill_supplies_agent_to_empty_worker_leaf() -> None:
    pe = _cordyceps_pe(executor_agent="")
    filled = _fill_empty_worker_agent(pe, "ralph-agent")

    assert _worker_agent_key_of(filled.execute.inner) == "ralph-agent", (
        "an empty worker leaf is filled with Ralph's agent (bare-leaf convenience)"
    )
    # The plan leaf's explicit agent is untouched either way.
    assert filled.plan.agent == "planner"


# The full Ralph wrapper path: filling descends through Ralph → PlanExecute →
# SelfVerifying → ReAct and respects the explicit executor leaf.
def test_fill_through_ralph_wrapper_respects_explicit_leaf() -> None:
    tree = RalphConfig(inner=_cordyceps_pe(executor_agent="executor"), agent="ralph-agent")
    # Mirror Ralph.run's call site: fill only when self.agent is set.
    inner = tree.inner if not tree.agent else _fill_empty_worker_agent(tree.inner, tree.agent)
    assert _worker_agent_key_of(inner.execute.inner) == "executor"


# ---------------------------------------------------------------------------
# Fix B: the plan phase honors the plan sub-strategy's declared PerLoop budget.
#
# The plan child is a ReAct ``PerLoop{4}`` leaf whose planner takes TWO turns to
# author the plan: turn 1 calls a tool, turn 2 emits the plan JSON. Under the
# old ``min(global, turns + 1)`` clamp the plan child was capped at ONE turn, so
# the planner never reached the JSON turn and the run failed. With the fix the
# plan child runs its full PerLoop{4}, authors the plan across 2 turns, and the
# run succeeds.
# ---------------------------------------------------------------------------


_PLAN_JSON = '{"tasks":["do the work"],"rationale":"authored after a tool call"}'


def _budget_test_config(agent: MockAgent, storage: StorageProvider) -> HarnessConfig:
    reg = ScriptedToolRegistry()
    # The planner's first-turn tool call gets one scripted success back, so the
    # ReAct loop continues into the second (JSON-emitting) turn.
    reg.push(ToolOutputSuccess(content="explored", truncated=False))
    return HarnessConfig(
        agent=agent,
        tool_registry=reg,
        sandbox=AllowAllSandbox(),
        context_manager=_StoringContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        storage=storage,
    )


class _StoringContextManager:
    """Stores appended user messages on the session and assembles them into the
    Context (same shape as the DAG test's stub), so the recorded agent observes
    the seeded directive."""

    async def assemble(self, session: SessionState, task: object) -> object:
        from spore_core.agent import Context
        from spore_core.model import ModelParams

        return Context(messages=list(session.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: object) -> None:
        return None

    async def append_user_message(self, session: SessionState, text: str) -> None:
        from spore_core.model import Message, Role, TextContent

        session.messages.append(Message(role=Role.USER, content=TextContent(text=text)))

    async def append_assistant_message(self, session: SessionState, message: object) -> None:
        session.messages.append(message)  # type: ignore[arg-type]

    def should_compact(self, session: SessionState) -> bool:
        return False


def _budget_tree() -> PlanExecuteConfig:
    """A PlanExecute whose plan child is a ReAct ``PerLoop{4}`` leaf and whose
    execute child is a bare ReAct leaf (drops the Ralph/SelfVerifying wrappers so
    the single MockAgent cursor maps 1:1 to the planner+worker turns)."""
    return PlanExecuteConfig(
        plan=ReactConfig(
            budget=BudgetPolicyPerLoop(value=4),
            agent="",
            toolset="",
            output="",
        ),
        execute=ReactConfig.per_loop(4),
    )


async def test_plan_phase_honors_declared_perloop_budget() -> None:
    storage = StorageProvider.single(InMemoryStorageProvider())
    agent = MockAgent(AgentId("test"))
    usage = TokenUsage(input_tokens=1, output_tokens=1)

    # Planner turn 1: a tool call (cannot finish the plan this turn).
    agent.push(
        ToolCallRequested(
            calls=[ToolCall(id="c1", name="explore", input={})],
            usage=usage,
        )
    )
    # Planner turn 2: emit the plan JSON (needs the declared budget > 1).
    agent.push(FinalResponse(content=_PLAN_JSON, usage=usage))
    # Worker turn: complete the single task.
    agent.push(FinalResponse(content="did the work", usage=usage))

    h = StandardHarness(_budget_test_config(agent, storage))
    task = Task.new(
        "build something",
        SessionId("plan-budget"),
        _budget_tree(),
        budget=BudgetLimits(max_turns=64),
    )
    r = await h.run(HarnessRunOptions(task))

    # Pre-fix: the plan child is clamped to 1 turn, never reaches the JSON turn,
    # and the run fails. With the fix the plan child uses its PerLoop{4} budget.
    assert isinstance(r, RunResultSuccess), f"expected Success, got {r!r}"
    # The planner authored across two turns (tool call + JSON) and the worker ran
    # once: three agent calls total.
    assert agent.call_count == 3, (
        f"planner not clamped to one turn — expected 3 total agent calls, "
        f"got {agent.call_count}"
    )
