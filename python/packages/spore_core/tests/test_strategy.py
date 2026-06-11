"""Tests for the recursive ``LoopStrategy`` config newtypes + ``RunStrategy`` +
``StrategyRef`` (issue #119, Composable Execution Part A).

``LoopStrategy`` is a closed, recursive discriminated union of config newtypes
(``ReactConfig`` leaf + combinators). Its wire format, and ``StrategyRef``'s,
must be byte-identical across Rust / TS / Python / Go. ``fixtures/strategy/`` is
the shared ground-truth corpus for that contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from spore_core.harness import (
    BudgetPolicyPerLoop,
    BudgetPolicyUnlimited,
    ChildPausedState,
    ExecutionContext,
    HillClimbingConfig,
    LoopStrategy,
    PausedState,
    PlanExecuteConfig,
    RalphConfig,
    ReactConfig,
    SelfVerifyingConfig,
    StrategyRef,
    StrategyRefBuiltIn,
    StrategyRefCustom,
    run_strategy,
)

_STRATEGY = TypeAdapter(LoopStrategy)
_REF = TypeAdapter(StrategyRef)


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "fixtures").is_dir() and (parent / "rust").is_dir():
            return parent
    raise RuntimeError("could not locate spore-core repo root from test file")


def _strategy_fixtures() -> Path:
    return _repo_root() / "fixtures/strategy"


def _compact(raw: str) -> str:
    """The canonical compact wire form for a (possibly pretty-printed) blob."""
    return json.dumps(json.loads(raw), separators=(",", ":"))


def _cordyceps_tree() -> LoopStrategy:
    """``Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]`` — the canonical tree."""
    return RalphConfig(
        inner=PlanExecuteConfig(
            plan=ReactConfig(
                budget=BudgetPolicyPerLoop(value=12),
                agent="planner",
                toolset="plan-tools",
                # A.5 (#124): the structured `plan` slot requires an output schema.
                output="plan-schema",
            ),
            execute=SelfVerifyingConfig(
                inner=ReactConfig(
                    budget=BudgetPolicyPerLoop(value=12),
                    agent="executor",
                    toolset="exec-tools",
                    # A.5 (#124): the structured `worker` slot requires an output
                    # schema.
                    output="worker-schema",
                ),
                evaluator="exec-evaluator",
            ),
        ),
        agent="ralph-agent",
    )


# ---------------------------------------------------------------------------
# Per-variant round-trip + exact serialized bytes
# ---------------------------------------------------------------------------


def test_react_leaf_roundtrip_and_tag() -> None:
    """The ReAct leaf tag is ``"react"`` (NOT ``"re_act"``); ``output`` is
    OMITTED from the wire when ``None``."""
    react = ReactConfig(budget=BudgetPolicyPerLoop(value=8), agent="a", toolset="t")
    wire = _STRATEGY.dump_json(react).decode()
    # #129: ``behavior`` ALWAYS serializes (default ``escalate``), CANONICAL
    # POSITION immediately after ``budget``.
    assert wire == (
        '{"kind":"react","budget":{"kind":"per_loop","value":8},'
        '"behavior":{"kind":"escalate"},"agent":"a","toolset":"t"}'
    )
    assert '"re_act"' not in wire
    assert "output" not in wire
    back = _STRATEGY.validate_json(wire)
    assert back == react


def test_react_output_present_when_set() -> None:
    react = ReactConfig(budget=BudgetPolicyUnlimited(), agent="a", toolset="t", output="schema-x")
    wire = _STRATEGY.dump_json(react).decode()
    assert '"output":"schema-x"' in wire
    assert _STRATEGY.validate_json(wire) == react


def test_plan_execute_roundtrip_omits_plan_model() -> None:
    pe = PlanExecuteConfig(plan=ReactConfig.per_loop(1), execute=ReactConfig.per_loop(2))
    wire = _STRATEGY.dump_json(pe).decode()
    assert "plan_model" not in wire
    assert _STRATEGY.validate_json(wire) == pe


def test_self_verifying_roundtrip() -> None:
    sv = SelfVerifyingConfig(inner=ReactConfig.per_loop(3), evaluator="judge")
    back = _STRATEGY.validate_json(_STRATEGY.dump_json(sv))
    assert back == sv


def test_ralph_roundtrip() -> None:
    r = RalphConfig(inner=ReactConfig.per_loop(3), agent="ralph")
    back = _STRATEGY.validate_json(_STRATEGY.dump_json(r))
    assert back == r


def test_hill_climbing_roundtrip() -> None:
    hc = HillClimbingConfig(
        inner=ReactConfig.per_loop(3),
        direction="maximize",
        max_stagnation=5,
        revert_on_no_improvement=True,
        min_improvement_delta=0.25,
        evaluator="metric",
    )
    wire = _STRATEGY.dump_json(hc).decode()
    assert '{"kind":"hill_climbing","inner":{"kind":"react",' in wire
    assert _STRATEGY.validate_json(wire) == hc


# ---------------------------------------------------------------------------
# Handle types round-trip as bare strings
# ---------------------------------------------------------------------------


def test_handles_roundtrip_as_bare_strings() -> None:
    """``AgentRef`` / ``ToolsetRef`` / ``SchemaRef`` are bare JSON strings on the
    wire (transparent)."""
    react = ReactConfig(
        budget=BudgetPolicyPerLoop(value=1),
        agent="my-agent",
        toolset="my-tools",
        output="my-schema",
    )
    data = json.loads(_STRATEGY.dump_json(react))
    assert data["agent"] == "my-agent"
    assert data["toolset"] == "my-tools"
    assert data["output"] == "my-schema"


# ---------------------------------------------------------------------------
# Recursive cordyceps tree round-trip + exact compact bytes
# ---------------------------------------------------------------------------


def test_cordyceps_tree_byte_identity() -> None:
    tree = _cordyceps_tree()
    raw = (_strategy_fixtures() / "cordyceps_tree.json").read_text()
    assert _STRATEGY.dump_json(tree).decode() == _compact(raw)
    assert _STRATEGY.validate_json(raw) == tree


def test_cordyceps_tree_fixture_replay() -> None:
    raw = (_strategy_fixtures() / "cordyceps_tree.json").read_text()
    model = _STRATEGY.validate_json(raw)
    assert _STRATEGY.dump_json(model).decode() == _compact(raw)
    assert json.loads(_STRATEGY.dump_json(model)) == json.loads(raw)
    # The parse really is Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]].
    assert isinstance(model, RalphConfig)
    assert isinstance(model.inner, PlanExecuteConfig)
    assert isinstance(model.inner.plan, ReactConfig)
    assert isinstance(model.inner.execute, SelfVerifyingConfig)
    assert isinstance(model.inner.execute.inner, ReactConfig)


# ---------------------------------------------------------------------------
# StrategyRef — BuiltIn / Custom (adjacently tagged on kind/value)
# ---------------------------------------------------------------------------


def test_strategy_ref_builtin_roundtrip() -> None:
    ref = StrategyRefBuiltIn(value=_cordyceps_tree())
    wire = _REF.dump_json(ref).decode()
    assert wire.startswith('{"kind":"built_in","value":{"kind":"ralph"')
    back = _REF.validate_json(wire)
    assert back == ref


def test_strategy_ref_custom_roundtrip() -> None:
    ref = StrategyRefCustom(value="my-harness::DoubleVerify")
    wire = _REF.dump_json(ref).decode()
    assert wire == '{"kind":"custom","value":"my-harness::DoubleVerify"}'
    back = _REF.validate_json(wire)
    assert isinstance(back, StrategyRefCustom)
    assert back.value == "my-harness::DoubleVerify"


def test_strategy_ref_fixture_replay() -> None:
    raw = json.loads((_strategy_fixtures() / "strategy_ref.json").read_text())
    for key in ("built_in", "custom"):
        model = _REF.validate_python(raw[key])
        assert json.loads(_REF.dump_json(model)) == raw[key]


# ---------------------------------------------------------------------------
# RunStrategy dispatch — every per-variant body returns a TYPED Failed (never
# raises) when the ExecutionContext has no StrategyExecutor wired (#124).
# Mirrors the Rust `run_without_executor_is_typed_failure_not_panic` test.
# ---------------------------------------------------------------------------


async def test_run_without_executor_is_typed_failure_not_panic() -> None:
    from spore_core.execution_registry import ExecutionRegistry
    from spore_core.harness import StrategyOutcomeFailed, Task, new_session_id

    for strategy in (
        ReactConfig.per_loop(1),
        PlanExecuteConfig.simple(),
        SelfVerifyingConfig.simple(),
        RalphConfig.simple(),
        HillClimbingConfig(
            inner=ReactConfig.per_loop(1),
            direction="minimize",
            max_stagnation=1,
            revert_on_no_improvement=False,
            min_improvement_delta=0.0,
            evaluator="m",
        ),
    ):
        cx = ExecutionContext(registry=ExecutionRegistry.empty())
        # The leaf reads scratch.task; set it so the failure is the missing
        # executor, not a missing task.
        cx.scratch.task = Task.new("t", new_session_id(), ReactConfig.per_loop(1))
        outcome = await run_strategy(strategy, cx)
        assert isinstance(outcome, StrategyOutcomeFailed)


# ---------------------------------------------------------------------------
# PausedState / ChildPausedState carry a cordyceps task.loop_strategy
# ---------------------------------------------------------------------------


def test_paused_state_fixture_roundtrip() -> None:
    raw = (_strategy_fixtures() / "paused_state.json").read_text()
    state = PausedState.model_validate_json(raw)
    assert isinstance(state.task.loop_strategy, RalphConfig)
    assert json.loads(state.model_dump_json()) == json.loads(raw)


def test_child_paused_state_fixture_roundtrip() -> None:
    raw = (_strategy_fixtures() / "child_paused_state.json").read_text()
    state = ChildPausedState.model_validate_json(raw)
    assert isinstance(state.task.loop_strategy, RalphConfig)
    assert json.loads(state.model_dump_json()) == json.loads(raw)
