"""Registry-validation test for example 08 (mirrors the Rust reference's
``registry_validates`` test).

AC: the composed ``PlanExecute(plan: ReAct{plan-schema}, execute: ReAct)`` tree
passes :meth:`ExecutionRegistry.validate` — the plan slot's output schema
resolves and the structured-slot contract is satisfied. The leaves use the EMPTY
agent/toolset handles that ``HarnessBuilder.build`` default-fills at build; here
we mirror that fill (empty-key agent + toolset) so the standalone registry
validates exactly as the assembled harness would.
"""

from __future__ import annotations

from spore_core import (
    AgentId,
    EmptyToolRegistry,
    ModelAgent,
    OllamaModelInterface,
    SessionId,
    Task,
)

from main import build_registry, plan_execute_strategy


def test_registry_validates() -> None:
    model = OllamaModelInterface.with_base_url("gemma4:e4b", "http://localhost:11434")
    registry = (
        build_registry()
        .into_builder()
        .fill_default_agent(ModelAgent(AgentId("default"), model))
        .fill_default_toolset(EmptyToolRegistry())
        .build()
    )
    task = Task.new("decompose and execute", SessionId("sess-08"), plan_execute_strategy())
    # ``validate`` returns ``None`` on success and raises on an unresolved handle.
    assert registry.validate(task) is None
