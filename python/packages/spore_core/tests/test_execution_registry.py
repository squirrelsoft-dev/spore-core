"""Tests for the :class:`ExecutionRegistry` (issue #120, Composable Execution
A.3).

The registry resolves the serializable strategy handles (``AgentRef`` /
``ToolsetRef`` / ``SchemaRef``) and ``StrategyRef.Custom`` keys carried in a
:class:`Task`'s strategy tree to concrete runtime collaborators. Trait objects
never enter the serialized ``Task`` — only string handles do — so on resume the
tree is reconstructed and every handle is re-resolved with no reconfiguration.

Covers every rule pinned in #120:
- unresolved handle → startup error before the first turn
- resume: serde round-trip a Task with refs, re-resolve all → all present
- missing ``StrategyRef.Custom`` key → recoverable ``StrategyNotFound``, no crash
- ``register_strategy`` then ``resolve_strategy(Custom(key))`` → ok
- EscalationMode present/selectable/readable on config
- each ``resolve_*`` happy + miss; tree-walk over ``cordyceps_tree.json``
- fixture-replay: round-trip ``fixtures/harness/registry_errors.json``
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
    EmptyToolRegistry,
    EscalationModeAutonomous,
    EscalationModeSurfaceToHuman,
    ExecutionRegistry,
    FinalResponse,
    HaltReasonConfigurationError,
    HarnessBuilder,
    HarnessError,
    HarnessErrorException,
    HarnessErrorStrategyNotFound,
    HarnessErrorUnresolvedHandle,
    HarnessRunOptions,
    MockAgent,
    NoopContextManager,
    ReactConfig,
    RunResultFailure,
    RunResultSuccess,
    SchemaRef,
    SessionId,
    StrategyOutcomeComplete,
    StrategyRefBuiltIn,
    StrategyRefCustom,
    Task,
    TokenUsage,
)
from spore_core.execution_registry import (
    StrategyResolutionBuiltIn,
    StrategyResolutionCustom,
)
from spore_core.harness import (
    BudgetPolicyPerLoop,
    ExecutionContext,
    LoopStrategy,
)

_STRATEGY = TypeAdapter(LoopStrategy)
_HARNESS_ERROR = TypeAdapter(HarnessError)
_TASK = TypeAdapter(Task)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "fixtures").is_dir() and (parent / "rust").is_dir():
            return parent
    raise RuntimeError("could not locate spore-core repo root from test file")


class _StubStrategy:
    """A minimal custom :class:`RunStrategy` (duck-typed). Never invoked in
    these tests — registry resolution is pure lookup."""

    async def run(self, cx: ExecutionContext) -> StrategyOutcomeComplete:
        return StrategyOutcomeComplete("")


def _agent() -> MockAgent:
    return MockAgent(AgentId("stub"))


def _react_leaf(agent: str, toolset: str, output: str | None = None) -> ReactConfig:
    return ReactConfig(
        budget=BudgetPolicyPerLoop(value=4),
        agent=agent,
        toolset=toolset,
        output=output,
    )


def _fully_wired_registry() -> ExecutionRegistry:
    return (
        ExecutionRegistry.builder()
        .agent("a1", _agent())
        .toolset("t1", EmptyToolRegistry())
        .schema("s1", {"type": "object"})
        .verifier("v1", object())
        .build()
    )


def _cordyceps_tree() -> LoopStrategy:
    raw = (_repo_root() / "fixtures/strategy/cordyceps_tree.json").read_text()
    return _STRATEGY.validate_json(raw)


# ---------------------------------------------------------------------------
# resolve_* happy path + miss
# ---------------------------------------------------------------------------


def test_resolve_each_happy_and_miss() -> None:
    reg = _fully_wired_registry()

    assert reg.resolve_agent("a1") is not None
    assert reg.resolve_agent("nope") is None

    assert reg.resolve_toolset("t1") is not None
    assert reg.resolve_toolset("nope") is None

    assert reg.resolve_schema("s1") is not None
    assert reg.resolve_schema("nope") is None

    assert reg.resolve_verifier("v1") is not None
    assert reg.resolve_verifier("nope") is None


def test_empty_registry_is_empty() -> None:
    assert ExecutionRegistry.empty().is_empty()
    reg = ExecutionRegistry.builder().agent("a1", _agent()).build()
    assert not reg.is_empty()


# ---------------------------------------------------------------------------
# register_strategy + resolve_strategy(Custom)
# ---------------------------------------------------------------------------


def test_register_then_resolve_custom_strategy() -> None:
    reg = ExecutionRegistry.empty()
    strat = _StubStrategy()
    reg.register_strategy("mine::Custom", strat)

    res = reg.resolve_strategy(StrategyRefCustom(value="mine::Custom"))
    assert isinstance(res, StrategyResolutionCustom)
    assert res.strategy is strat


def test_resolve_builtin_strategy_borrows_tree() -> None:
    reg = ExecutionRegistry.empty()
    leaf = _react_leaf("a1", "t1")
    res = reg.resolve_strategy(StrategyRefBuiltIn(value=leaf))
    assert isinstance(res, StrategyResolutionBuiltIn)
    assert isinstance(res.strategy, ReactConfig)


# ---------------------------------------------------------------------------
# missing custom key → recoverable StrategyNotFound, no crash
# ---------------------------------------------------------------------------


def test_missing_custom_key_is_recoverable_strategy_not_found() -> None:
    reg = ExecutionRegistry.empty()
    with pytest.raises(HarnessErrorException) as ei:
        reg.resolve_strategy(StrategyRefCustom(value="absent"))
    assert ei.value.error == HarnessErrorStrategyNotFound(key="absent")
    assert ei.value.error.message() == "custom strategy not found: absent"


# ---------------------------------------------------------------------------
# validate() unresolved handle → UnresolvedHandle
# ---------------------------------------------------------------------------


def test_validate_unresolved_agent_handle() -> None:
    reg = ExecutionRegistry.empty()
    task = Task.new("do it", SessionId("s1"), _react_leaf("missing-agent", "t1"))
    with pytest.raises(HarnessErrorException) as ei:
        reg.validate(task)
    assert ei.value.error == HarnessErrorUnresolvedHandle(handle_kind="agent", key="missing-agent")


def test_validate_unresolved_toolset_handle() -> None:
    reg = ExecutionRegistry.builder().agent("a1", _agent()).build()
    task = Task.new("do it", SessionId("s1"), _react_leaf("a1", "missing-tools"))
    with pytest.raises(HarnessErrorException) as ei:
        reg.validate(task)
    assert ei.value.error == HarnessErrorUnresolvedHandle(
        handle_kind="toolset", key="missing-tools"
    )


def test_validate_unresolved_schema_handle() -> None:
    reg = (
        ExecutionRegistry.builder().agent("a1", _agent()).toolset("t1", EmptyToolRegistry()).build()
    )
    task = Task.new("do it", SessionId("s1"), _react_leaf("a1", "t1", output="missing-schema"))
    with pytest.raises(HarnessErrorException) as ei:
        reg.validate(task)
    assert ei.value.error == HarnessErrorUnresolvedHandle(
        handle_kind="schema", key="missing-schema"
    )


def test_validate_happy_path_react() -> None:
    reg = _fully_wired_registry()
    task = Task.new("ok", SessionId("s1"), _react_leaf("a1", "t1", output="s1"))
    # No raise == valid.
    reg.validate(task)


# ---------------------------------------------------------------------------
# tree-walk over the nested cordyceps fixture tree
# ---------------------------------------------------------------------------


def test_validate_tree_walk_reports_first_unresolved_in_nested_tree() -> None:
    # The cordyceps tree references agents planner/executor/ralph-agent, toolsets
    # plan-tools/exec-tools, schema exec-evaluator. An empty registry must report
    # the FIRST unresolved handle (depth-first: ralph inner → plan_execute → plan
    # react → agent "planner").
    reg = ExecutionRegistry.empty()
    task = Task.new("nested", SessionId("s1"), _cordyceps_tree())
    with pytest.raises(HarnessErrorException) as ei:
        reg.validate(task)
    assert ei.value.error == HarnessErrorUnresolvedHandle(handle_kind="agent", key="planner")


def test_validate_tree_walk_passes_when_fully_wired() -> None:
    reg = (
        ExecutionRegistry.builder()
        .agent("planner", _agent())
        .agent("executor", _agent())
        .agent("ralph-agent", _agent())
        .toolset("plan-tools", EmptyToolRegistry())
        .toolset("exec-tools", EmptyToolRegistry())
        # #124 Q1: the SelfVerifying ``evaluator`` resolves as a VERIFIER key.
        .verifier("exec-evaluator", object())
        # A.5 (#124): the structured plan/worker slots now carry output schemas.
        .schema("plan-schema", {})
        .schema("worker-schema", {})
        .build()
    )
    task = Task.new("nested", SessionId("s1"), _cordyceps_tree())
    reg.validate(task)


# ---------------------------------------------------------------------------
# resume: round-trip a Task through serde, re-resolve all
# ---------------------------------------------------------------------------


def test_resume_reresolves_all_handles_after_serde_roundtrip() -> None:
    leaf = _react_leaf("a1", "t1", output="s1")
    task = Task.new("resume me", SessionId("s1"), leaf)

    wire = _TASK.dump_json(task)
    restored = _TASK.validate_json(wire)

    # Fresh registry built independently (as on resume) re-resolves all — no
    # reconfiguration of the restored Task required.
    reg = _fully_wired_registry()
    reg.validate(restored)

    assert isinstance(restored.loop_strategy, ReactConfig)
    c = restored.loop_strategy
    assert reg.resolve_agent(c.agent) is not None
    assert reg.resolve_toolset(c.toolset) is not None
    assert c.output is not None
    assert reg.resolve_schema(c.output) is not None


# ---------------------------------------------------------------------------
# builder: last-wins + into_builder round-trip
# ---------------------------------------------------------------------------


def test_builder_last_wins_on_duplicate_key() -> None:
    reg = ExecutionRegistry.builder().schema("s", {"v": 1}).schema("s", {"v": 2}).build()
    assert reg.resolve_schema(SchemaRef("s")) == {"v": 2}


def test_into_builder_preserves_entries() -> None:
    reg = ExecutionRegistry.builder().agent("a1", _agent()).build()
    reg2 = reg.into_builder().toolset("t1", EmptyToolRegistry()).build()
    assert reg2.resolve_agent("a1") is not None
    assert reg2.resolve_toolset("t1") is not None


# ---------------------------------------------------------------------------
# EscalationMode present/selectable/readable on config
# ---------------------------------------------------------------------------


def test_escalation_mode_defaults_to_surface_to_human() -> None:
    cfg = _builder().build_config()
    assert isinstance(cfg.escalation_mode, EscalationModeSurfaceToHuman)


def test_escalation_mode_is_selectable_and_readable() -> None:
    cfg = _builder().escalation_mode(EscalationModeAutonomous()).build_config()
    assert isinstance(cfg.escalation_mode, EscalationModeAutonomous)


def test_escalation_mode_serde_shape() -> None:
    assert EscalationModeSurfaceToHuman().model_dump() == {"kind": "surface_to_human"}
    assert EscalationModeAutonomous().model_dump() == {"kind": "autonomous"}


# ---------------------------------------------------------------------------
# registry wired onto HarnessConfig / HarnessBuilder
# ---------------------------------------------------------------------------


def _builder() -> HarnessBuilder:
    return HarnessBuilder(
        _agent(),
        EmptyToolRegistry(),
        AllowAllSandbox(),
        NoopContextManager(),
        AlwaysContinuePolicy(),
    )


def test_config_registry_folds_default_agent() -> None:
    # #124: the legacy single-collaborator fields are gone — ``HarnessConfig``
    # folds the builder's agent + toolset + default schema into the registry under
    # the DEFAULT empty key, so the single resolution path always has a worker to
    # resolve. Mirrors Rust's ``builder_folds_default_agent_into_registry``.
    cfg = _builder().build_config()
    assert not cfg.registry.is_empty()
    assert cfg.registry.resolve_agent("") is not None


def test_builder_register_convenience_setters() -> None:
    cfg = (
        _builder()
        .register_agent("a1", _agent())
        .register_toolset("t1", EmptyToolRegistry())
        .register_schema("s1", {})
        .register_verifier("v1", object())
        .register_strategy("c1", _StubStrategy())
        .build_config()
    )
    assert cfg.registry.resolve_agent("a1") is not None
    assert cfg.registry.resolve_toolset("t1") is not None
    assert cfg.registry.resolve_schema("s1") is not None
    assert cfg.registry.resolve_verifier("v1") is not None
    res = cfg.registry.resolve_strategy(StrategyRefCustom(value="c1"))
    assert isinstance(res, StrategyResolutionCustom)


# ---------------------------------------------------------------------------
# unresolved handle → startup error before the first turn (run entry)
# ---------------------------------------------------------------------------


async def test_run_entry_unresolved_handle_is_startup_failure() -> None:
    # A populated registry that does NOT resolve the task's handles fails at run
    # entry with ConfigurationError, before the agent takes any turn (the agent
    # has zero scripted turns — if it ran it would error).
    h = (
        HarnessBuilder(
            _agent(),
            EmptyToolRegistry(),
            AllowAllSandbox(),
            NoopContextManager(),
            AlwaysContinuePolicy(),
        )
        .register_agent("present", _agent())
        .build()
    )
    task = Task.new("do it", SessionId("s1"), _react_leaf("missing-agent", "present"))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonConfigurationError)
    assert r.reason.error == HarnessErrorUnresolvedHandle(handle_kind="agent", key="missing-agent")
    assert r.turns == 0


async def test_run_entry_default_folded_leaf_validates_and_runs() -> None:
    # #124: validation is now the SINGLE resolution path and ALWAYS runs (the old
    # "empty registry skips validation" gate is dropped). A bare default ReAct
    # leaf (empty handles) resolves against the defaults the builder folds into the
    # registry, so the run validates and proceeds.
    a = _agent()
    a.push(FinalResponse(content="done", usage=TokenUsage(input_tokens=1, output_tokens=1)))
    h = HarnessBuilder(
        a, EmptyToolRegistry(), AllowAllSandbox(), NoopContextManager(), AlwaysContinuePolicy()
    ).build()
    # The default empty-key leaf resolves via the folded default agent + toolset.
    task = Task.new("do it", SessionId("s1"), ReactConfig.per_loop(5))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "done"


# ---------------------------------------------------------------------------
# fixture replay: HarnessError variants round-trip byte-identically
# ---------------------------------------------------------------------------


def test_registry_errors_fixture_round_trips_byte_identical() -> None:
    raw = (_repo_root() / "fixtures/harness/registry_errors.json").read_text()
    doc = json.loads(raw)

    snf = _HARNESS_ERROR.validate_python(doc["strategy_not_found"])
    assert snf == HarnessErrorStrategyNotFound(key="my-harness::DoubleVerify")
    assert _HARNESS_ERROR.dump_python(snf, mode="json") == doc["strategy_not_found"]

    uh = _HARNESS_ERROR.validate_python(doc["unresolved_handle"])
    assert uh == HarnessErrorUnresolvedHandle(handle_kind="agent", key="planner")
    assert _HARNESS_ERROR.dump_python(uh, mode="json") == doc["unresolved_handle"]
