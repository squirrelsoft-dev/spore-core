"""Registry-validation test for example 09 (mirrors the Rust reference's
``registry_validates`` test).

AC: the composed ``SelfVerifying(inner: ReAct{worker-schema}, evaluator: "")``
tree validates — the worker slot's output schema resolves (structured-slot
contract) and the empty-handle evaluator resolves to the default-filled verifier.
The leaves use EMPTY agent/toolset/evaluator handles that ``HarnessBuilder.build``
default-fills at build; here we mirror that fill (empty-key agent + toolset +
verifier) so the standalone registry validates exactly as the assembled harness
would.
"""

from __future__ import annotations

from spore_core import (
    AgentId,
    EmptyToolRegistry,
    EvaluatorResponseVerifier,
    ModelAgent,
    OllamaModelInterface,
    SessionId,
    Task,
)

from main import build_registry, self_verifying_strategy


def test_registry_validates() -> None:
    model = OllamaModelInterface.with_base_url("gemma4:e4b", "http://localhost:11434")
    verifier = EvaluatorResponseVerifier(
        pass_pattern=r"(?im)^\s*PASS\s*$",
        fail_pattern=r"(?im)FAIL:\s*.+",
        max_iterations=3,
    )
    registry = (
        build_registry()
        .into_builder()
        .fill_default_agent(ModelAgent(AgentId("default"), model))
        .fill_default_toolset(EmptyToolRegistry())
        .fill_default_verifier(verifier)
        .build()
    )
    task = Task.new("draft and verify", SessionId("sess-09"), self_verifying_strategy(3))
    # ``validate`` returns ``None`` on success and raises on an unresolved handle.
    assert registry.validate(task) is None
