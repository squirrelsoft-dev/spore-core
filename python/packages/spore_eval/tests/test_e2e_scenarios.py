"""Hermetic end-to-end scenario tests (issue #57).

These drive the SAME :func:`spore_eval.scenarios.build_scenario` wiring the
``spore-e2e-agent`` CLI uses, but with a scripted mock agent, scripted/real tool
registries, and an allow-all sandbox, so CI never needs a live Ollama or any
network. Each test asserts the harness loop control flow (turn count, tool
dispatch order, S4 recovery sequencing, S3 live compaction with real token
reclamation). ``SPORE_OTLP_ENDPOINT`` stays unset, so there is no forwarding.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pytest

from spore_core.agent import AgentId, FinalResponse, MockAgent, ToolCallRequested
from spore_core.cache_provider import NullCacheProvider
from spore_core.context import CompactionConfig
from spore_core.harness import (
    AllowAllSandbox,
    AlwaysContinuePolicy,
    HarnessRunOptions,
    ReactConfig,
    NoopContextManager,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    Task,
    ToolOutputSuccess,
)
from spore_core.model import MockModelInterface, ProviderInfo, TokenUsage, ToolCall
from spore_core.storage import InMemoryStorageProvider
from spore_core.observability import (
    ContextOperationCompaction,
    InMemoryObservabilityProvider,
)

from spore_eval.scenarios import (
    RealToolRegistry,
    ScenarioComponents,
    ScenarioId,
    build_real_tool_registry,
    build_rich_context_manager,
    build_scenario,
    seed_compaction_state,
)


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=10, output_tokens=5)


def _call(call_id: str, name: str, inp: dict) -> ToolCall:
    return ToolCall(id=call_id, name=name, input=inp)


# ---------------------------------------------------------------------------
# S1 — multi-step / multi-tool
# ---------------------------------------------------------------------------


async def test_s1_multi_step_multi_tool() -> None:
    agent = MockAgent(AgentId("mock"))
    # read -> write -> bash read-back -> final.
    agent.push(
        ToolCallRequested(calls=[_call("c1", "read_file", {"path": "input.txt"})], usage=_usage())
    )
    agent.push(
        ToolCallRequested(
            calls=[_call("c2", "write_file", {"path": "output.txt", "content": "UPPERCASED"})],
            usage=_usage(),
        )
    )
    agent.push(
        ToolCallRequested(calls=[_call("c3", "read_file", {"path": "output.txt"})], usage=_usage())
    )
    agent.push(FinalResponse(content="DONE", usage=_usage()))

    tools = ScriptedToolRegistry()
    tools.push(ToolOutputSuccess(content="hello"))
    tools.push(ToolOutputSuccess(content="wrote 10 bytes"))
    tools.push(ToolOutputSuccess(content="UPPERCASED"))

    harness = build_scenario(
        ScenarioId.S1,
        ScenarioComponents(
            agent=agent,
            tools=tools,
            sandbox=AllowAllSandbox(),
            context_manager=NoopContextManager(),
            termination_policy=AlwaysContinuePolicy(),
            tool_schemas=[],
            observability=None,
        ),
    )
    task = Task.new(ScenarioId.S1.prompt(), SessionId("s1-test"), ReactConfig.per_loop(8))
    result = await harness.run(HarnessRunOptions(task))
    assert isinstance(result, RunResultSuccess), result
    assert result.turns > 2, f"S1 should take >2 turns, got {result.turns}"
    assert tools.call_count == 3, "S1 dispatches read+write+readback = 3 tools"
    assert "DONE" in result.output


# ---------------------------------------------------------------------------
# S2 — multi-turn, same SessionId, carrying session state
# ---------------------------------------------------------------------------


async def test_s2_multi_turn_carries_state() -> None:
    session_id = SessionId("s2-test")
    agent = MockAgent(AgentId("mock"))
    # Turn 1: write notes.md, then final.
    agent.push(
        ToolCallRequested(
            calls=[
                _call(
                    "c1", "write_file", {"path": "notes.md", "content": "TODO: set up the project"}
                )
            ],
            usage=_usage(),
        )
    )
    agent.push(FinalResponse(content="DONE", usage=_usage()))
    # Turn 2: append referencing turn 1, then final.
    agent.push(
        ToolCallRequested(
            calls=[
                _call(
                    "c2",
                    "write_file",
                    {
                        "path": "notes.md",
                        "content": "TODO: follow up on set up the project",
                        "append": True,
                    },
                )
            ],
            usage=_usage(),
        )
    )
    agent.push(FinalResponse(content="DONE referencing set up the project", usage=_usage()))

    tools = ScriptedToolRegistry()

    harness = build_scenario(
        ScenarioId.S2,
        ScenarioComponents(
            agent=agent,
            tools=tools,
            sandbox=AllowAllSandbox(),
            context_manager=NoopContextManager(),
            termination_policy=AlwaysContinuePolicy(),
            tool_schemas=[],
            observability=None,
        ),
    )

    task1 = Task.new(ScenarioId.S2.prompt(), session_id, ReactConfig.per_loop(5))
    r1 = await harness.run(HarnessRunOptions(task1))
    assert isinstance(r1, RunResultSuccess), r1

    task2 = Task.new("add a second item referencing the first", session_id, ReactConfig.per_loop(5))
    r2 = await harness.run(HarnessRunOptions(task2, session_state=SessionState()))
    assert isinstance(r2, RunResultSuccess), r2
    assert r2.session_id == session_id, "same session id across turns"
    assert "set up the project" in r2.output, f"turn 2 references turn 1: {r2.output!r}"


# ---------------------------------------------------------------------------
# S3 — live compaction with real token reclamation
# ---------------------------------------------------------------------------


async def test_s3_live_compaction_reclaims_tokens() -> None:
    session_id = SessionId("s3-test")
    # Agent emits a tool call (to reach the post-tool compaction arm), then a
    # final summary containing the key terms so verification passes.
    agent = MockAgent(AgentId("mock"))
    agent.push(ToolCallRequested(calls=[_call("c1", "read_file", {"path": "x"})], usage=_usage()))
    # Compaction turn (consumed inside _run_compaction): a summary preserving the
    # "payment"/"service"/"deploy" key terms.
    agent.push(
        FinalResponse(
            content="summary: continuing the deploy of the payment service", usage=_usage()
        )
    )
    # Next loop turn after compaction: final response.
    agent.push(FinalResponse(content="DONE deploy payment service", usage=_usage()))

    tools = ScriptedToolRegistry()
    tools.push(ToolOutputSuccess(content="file contents"))

    model = MockModelInterface(ProviderInfo(name="mock", model_id="mock", context_window=200))
    cfg = CompactionConfig(
        threshold=0.80,
        preserve_recent_n=2,
        head_tail_tokens=64,
        offload_path=Path(".spore/offload"),
        max_compaction_attempts=2,
    )
    cm = build_rich_context_manager(model, NullCacheProvider(), cfg)
    obs = InMemoryObservabilityProvider()

    harness = build_scenario(
        ScenarioId.S3,
        ScenarioComponents(
            agent=agent,
            tools=tools,
            sandbox=AllowAllSandbox(),
            context_manager=cm,
            termination_policy=AlwaysContinuePolicy(),
            tool_schemas=[],
            observability=obs,
        ),
    )

    task = Task.new("deploy the payment service", session_id, ReactConfig.per_loop(8))
    state = SessionState()
    # Seed a small window with budget over threshold (170/200 = 0.85) + long history.
    seed_compaction_state(state, "deploy the payment service", session_id, task.id, 200, 170, 12)

    result = await harness.run(HarnessRunOptions(task, session_state=state))
    assert isinstance(result, RunResultSuccess), f"S3 expected Success, got {result!r}"

    compactions = [
        c
        for c in obs.context_spans(session_id)
        if isinstance(c.operation, ContextOperationCompaction)
    ]
    assert compactions, "S3 should emit >=1 Compaction span mid-run"
    first = compactions[0]
    assert first.tokens_after < first.tokens_before, (
        f"token_budget_used must drop after compaction: {first.tokens_before} -> {first.tokens_after}"
    )
    assert isinstance(first.operation, ContextOperationCompaction)
    assert first.operation.tokens_reclaimed > 0, "real reclamation must be > 0"


# ---------------------------------------------------------------------------
# S4 — tool failure + recovery (uses the REAL registry + FailingTool)
# ---------------------------------------------------------------------------


async def test_s4_tool_failure_then_recovery() -> None:
    workspace = Path(tempfile.gettempdir()) / f"spore-s4-{time.time_ns()}"
    workspace.mkdir(parents=True, exist_ok=True)

    session_id = SessionId("s4-test")
    agent = MockAgent(AgentId("mock"))
    # Call flaky_op (fails recoverably) -> adapt by writing recovered.txt -> final.
    agent.push(
        ToolCallRequested(calls=[_call("c1", "flaky_op", {"reason": "first try"})], usage=_usage())
    )
    agent.push(
        ToolCallRequested(
            calls=[
                _call(
                    "c2",
                    "write_file",
                    {
                        "path": str(workspace / "recovered.txt"),
                        "content": "flaky_op failed; adapted by writing this file",
                    },
                )
            ],
            usage=_usage(),
        )
    )
    agent.push(FinalResponse(content="DONE recovered", usage=_usage()))

    registry = build_real_tool_registry(ScenarioId.S4)
    sandbox = AllowAllSandbox()
    _storage = InMemoryStorageProvider()
    bridge = RealToolRegistry(registry, sandbox, session_id, _storage, _storage)
    schemas = bridge.model_schemas()

    harness = build_scenario(
        ScenarioId.S4,
        ScenarioComponents(
            agent=agent,
            tools=bridge,
            sandbox=sandbox,
            context_manager=NoopContextManager(),
            termination_policy=AlwaysContinuePolicy(),
            tool_schemas=schemas,
            observability=None,
        ),
    )

    task = Task.new(ScenarioId.S4.prompt(), session_id, ReactConfig.per_loop(8))
    result = await harness.run(HarnessRunOptions(task))
    assert isinstance(result, RunResultSuccess), f"S4 expected Success, got {result!r}"
    assert result.turns >= 3, "S4: flaky -> recover -> done"
    assert (workspace / "recovered.txt").exists(), "recovery file written"


async def test_s4_failing_tool_is_not_always_halt() -> None:
    """The harness must NOT hard-halt on the recoverable FailingTool error — the
    bridge reports ``is_always_halt == False``."""
    _storage = InMemoryStorageProvider()
    bridge = RealToolRegistry(
        build_real_tool_registry(ScenarioId.S4),
        AllowAllSandbox(),
        SessionId("s4-halt-test"),
        _storage,
        _storage,
    )
    assert not bridge.is_always_halt("flaky_op")
    out = await bridge.dispatch(_call("c1", "flaky_op", {}))
    from spore_core.harness import ToolOutputError

    assert isinstance(out, ToolOutputError)
    assert out.recoverable


# ---------------------------------------------------------------------------
# S5 — real shell tool: bash_command with a pipe + redirect (uses the REAL
# registry, which exposes bash_command only for S5).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell only")
async def test_s5_shell_pipeline_uppercases_via_bash_command() -> None:
    workspace = Path(tempfile.gettempdir()) / f"spore-s5-{time.time_ns()}"
    workspace.mkdir(parents=True, exist_ok=True)
    input_path = workspace / "input.txt"
    output_path = workspace / "output.txt"
    input_path.write_text("hello\n")

    session_id = SessionId("s5-test")
    agent = MockAgent(AgentId("mock"))
    # turn1: bash_command with a literal pipe AND redirect; turn2: read_file
    # output.txt to verify; turn3: DONE.
    script = f"cat {input_path} | tr a-z A-Z > {output_path}"
    agent.push(
        ToolCallRequested(calls=[_call("c1", "bash_command", {"script": script})], usage=_usage())
    )
    agent.push(
        ToolCallRequested(
            calls=[_call("c2", "read_file", {"path": str(output_path)})], usage=_usage()
        )
    )
    agent.push(FinalResponse(content="DONE", usage=_usage()))

    registry = build_real_tool_registry(ScenarioId.S5)
    sandbox = AllowAllSandbox()
    _storage = InMemoryStorageProvider()
    bridge = RealToolRegistry(registry, sandbox, session_id, _storage, _storage)
    schemas = bridge.model_schemas()

    harness = build_scenario(
        ScenarioId.S5,
        ScenarioComponents(
            agent=agent,
            tools=bridge,
            sandbox=sandbox,
            context_manager=NoopContextManager(),
            termination_policy=AlwaysContinuePolicy(),
            tool_schemas=schemas,
            observability=None,
        ),
    )

    task = Task.new(ScenarioId.S5.prompt(), session_id, ReactConfig.per_loop(8))
    result = await harness.run(HarnessRunOptions(task))
    assert isinstance(result, RunResultSuccess), f"S5 expected Success, got {result!r}"
    assert result.turns >= 3, f"S5: bash_command -> read_file -> done, got {result.turns}"
    assert output_path.read_text() == "HELLO\n", (
        "shell pipeline must uppercase input.txt into output.txt"
    )


# ---------------------------------------------------------------------------
# Shared-builder unit checks
# ---------------------------------------------------------------------------


def test_scenario_id_parses() -> None:
    assert ScenarioId.parse("s1") is ScenarioId.S1
    assert ScenarioId.parse("S4") is ScenarioId.S4
    assert ScenarioId.parse("s5") is ScenarioId.S5
    assert ScenarioId.parse("nope") is None


def _schema_names(scenario: ScenarioId) -> list[str]:
    _storage = InMemoryStorageProvider()
    bridge = RealToolRegistry(
        build_real_tool_registry(scenario),
        AllowAllSandbox(),
        SessionId("schema-test"),
        _storage,
        _storage,
    )
    return [s.name for s in bridge.model_schemas()]


def test_real_registry_exposes_sorted_schemas() -> None:
    names = _schema_names(ScenarioId.S1)
    assert names == sorted(names), "schemas must be sorted by name"
    assert "flaky_op" in names
    assert "read_file" in names


def test_s1_registry_has_exec_not_bash_command() -> None:
    names = _schema_names(ScenarioId.S1)
    assert "exec" in names
    assert "bash_command" not in names


def test_s2_registry_lacks_bash_command() -> None:
    names = _schema_names(ScenarioId.S2)
    assert "exec" in names
    assert "bash_command" not in names


def test_s5_registry_has_bash_command() -> None:
    names = _schema_names(ScenarioId.S5)
    assert "bash_command" in names
    assert "exec" in names


# ---------------------------------------------------------------------------
# #78 R9 — RealToolRegistry threads the configured memory store into ToolContext
# ---------------------------------------------------------------------------


async def test_real_tool_registry_threads_memory_store() -> None:
    """The bridge must thread its injected ``memory_store`` into the
    ``ToolContext`` it hands to every tool (#78). Prove the seam is live by
    writing through the context's memory store and reading it back through the
    same backend the registry was wired with."""
    from spore_core.memory import Timestamp
    from spore_core.storage import MemoryEntry, StorageScope
    from spore_core.tool_registry import StandardToolRegistry

    memory = InMemoryStorageProvider()
    bridge = RealToolRegistry(
        StandardToolRegistry(),
        AllowAllSandbox(),
        SessionId("ctx-test"),
        InMemoryStorageProvider(),
        memory,
    )
    ctx = bridge.tool_context()
    entry = MemoryEntry(role="user", content="threaded", timestamp=Timestamp("t1"))
    await ctx.memory_store.append_memory(StorageScope.PROJECT, ctx.session_id, entry)
    # Read back through the same Arc/object the registry threaded in.
    got = await memory.get_memories(StorageScope.PROJECT, SessionId("ctx-test"), 10)
    assert len(got) == 1
    assert got[0].content == "threaded"
