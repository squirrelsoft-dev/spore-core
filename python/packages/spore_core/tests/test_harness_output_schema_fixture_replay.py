"""Fixture-replay tests for output-schema delivery + enforcement (issue #139).

Replays the three SHARED fixtures under
``fixtures/model_responses/harness/output_schema_{accept,retry,fail}.jsonl``
through a :class:`StandardHarness` with output-schema enforcement ON. These are
GROUND TRUTH: the SAME fixtures drive the Rust / TypeScript / Go replay tests, so
the Python implementation must produce the SAME terminal outcome. Never edit the
fixtures to make a failing implementation pass (see ``fixtures/README.md``).

- accept: one valid terminal ⇒ Success on turn 1.
- retry:  invalid then valid ⇒ Success on turn 2 (one retry consumed).
- fail:   three invalid terminals (N == 2 ⇒ 3 attempts) ⇒ OutputSchemaViolation
          (distinct from budget; turns == 3 < budget).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    EscalationModeAutonomous,
    HaltReasonOutputSchemaViolation,
    HarnessConfig,
    HarnessRunOptions,
    ProviderInfo,
    ReactConfig,
    ReplayModelInterface,
    RunResultFailure,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    StandardCompactionAdapter,
    StandardContextManager,
    StandardHarness,
    Task,
)
from spore_core.agent import ModelAgent
from spore_core.context import CompactionConfig
from spore_core.execution_registry import ExecutionRegistry


def _fixture_path(name: str) -> Path:
    here = Path(__file__).resolve()
    return (
        here.parents[4] / "fixtures" / "model_responses" / "harness" / f"output_schema_{name}.jsonl"
    )


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["status", "count"],
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "count": {"type": "integer"},
        },
    }


class _StubModel:
    async def call(self, request: object) -> object:  # pragma: no cover - unused
        raise NotImplementedError

    async def call_streaming(self, request: object) -> object:  # pragma: no cover - unused
        raise NotImplementedError

    async def count_tokens(self, request: object) -> int:
        return 0

    def provider(self) -> ProviderInfo:
        return ProviderInfo(name="stub", model_id="stub", context_window=200_000)


def _rich_adapter() -> StandardCompactionAdapter:
    cfg = CompactionConfig(
        threshold=0.80,
        preserve_recent_n=2,
        head_tail_tokens=64,
        max_compaction_attempts=2,
    )
    return StandardCompactionAdapter(StandardContextManager(_StubModel(), compaction=cfg))


def _harness(name: str) -> StandardHarness:
    jsonl = _fixture_path(name).read_text()
    replay = ReplayModelInterface.from_jsonl(
        jsonl,
        ProviderInfo(name="ollama", model_id="fixture", context_window=200_000),
    )
    agent = ModelAgent(AgentId("fixture-agent"), replay)
    config = HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=_rich_adapter(),
        termination_policy=AlwaysContinuePolicy(),
        enforce_output_schemas=True,
        output_schema_max_retries=2,
        registry=ExecutionRegistry.builder().schema("", _schema()).build(),
        escalation_mode=EscalationModeAutonomous(),
    )
    return StandardHarness(config)


def _leaf(budget: int = 50) -> Task:
    return Task.new(
        "produce a status report",
        SessionId("output-schema-replay"),
        ReactConfig(
            budget=ReactConfig.per_loop(budget).budget,
            agent="",
            toolset="",
            output="",
        ),
    )


async def test_output_schema_accept_fixture() -> None:
    h = _harness("accept")
    r = await h.run(HarnessRunOptions(_leaf(50)))
    assert isinstance(r, RunResultSuccess), f"expected Success, got {r!r}"
    assert r.output == '{"status":"ok","count":3}'
    assert r.turns == 1, "valid on turn 1, no retry"


async def test_output_schema_retry_fixture() -> None:
    h = _harness("retry")
    r = await h.run(HarnessRunOptions(_leaf(50)))
    assert isinstance(r, RunResultSuccess), f"expected Success, got {r!r}"
    assert r.output == '{"status":"ok","count":3}'
    assert r.turns == 2, "one retry consumed (invalid then valid)"


async def test_output_schema_fail_fixture() -> None:
    h = _harness("fail")
    r = await h.run(HarnessRunOptions(_leaf(50)))
    assert isinstance(r, RunResultFailure), f"expected Failure, got {r!r}"
    assert isinstance(r.reason, HaltReasonOutputSchemaViolation), (
        f"expected OutputSchemaViolation, got {r.reason!r}"
    )
    assert r.reason.attempts == 3, "1 + N == 1 + 2 attempts"
    assert r.reason.last_error == 'Missing required property "status".'
    assert r.turns == 3, "exactly 1 + N turns"
    assert r.turns < 50, "budget NOT exhausted (distinct from budget)"
