"""Registry-validation test for example 10 (mirrors the Rust reference's
``registry_validates`` test).

AC: the composed ``HillClimbing(inner: ReAct{propose-schema}, evaluator: "")``
tree validates — the propose slot's output schema resolves (structured-slot
contract) and the empty-handle evaluator resolves to the default-filled metric
evaluator. The leaves use EMPTY agent/toolset/evaluator handles that
``HarnessBuilder.build`` default-fills at build; here we mirror that fill
(empty-key agent + toolset + metric evaluator) so the standalone registry
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

from main import (
    PER_ITER_BUDGET,
    ReadmeQualityEvaluator,
    build_registry,
    hill_climbing_strategy,
)


def test_registry_validates() -> None:
    model = OllamaModelInterface.with_base_url("gemma4:e4b", "http://localhost:11434")
    metric = ReadmeQualityEvaluator(model)
    registry = (
        build_registry()
        .into_builder()
        .fill_default_agent(ModelAgent(AgentId("default"), model))
        .fill_default_toolset(EmptyToolRegistry())
        .fill_default_metric_evaluator(metric)
        .build()
    )
    task = Task.new("refine the README", SessionId("sess-10"), hill_climbing_strategy(PER_ITER_BUDGET))
    # ``validate`` returns ``None`` on success and raises on an unresolved handle.
    assert registry.validate(task) is None
