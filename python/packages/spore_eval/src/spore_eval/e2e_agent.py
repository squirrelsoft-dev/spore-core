"""End-to-end CLI agent harness + scenario suite (issue #57).

One shared runnable entry point that drives the *complete* harness loop against
a real local model (Ollama) through a ``HarnessBuilder``-assembled harness with
real tools (read/write/list/bash + a deliberately-failing tool), the
``StandardCompactionAdapter``, and the durable-outbox observability provider.
The scenario is selected by a CLI arg.

Scenarios
---------
* ``s1`` — multi-step / multi-tool: read input.txt → uppercase → write
  output.txt → read back + confirm.
* ``s2`` — multi-turn: run twice with the same ``SessionId``, carrying session
  state across turns.
* ``s3`` — live compaction: a seeded small window + long history fires the
  compaction adapter mid-run; the token-accounting fix lets it compact,
  continue, and compact again.
* ``s4`` — tool failure + recovery: call ``flaky_op`` (recoverable error), then
  write recovered.txt explaining the adaptation.
* ``s5`` — real shell tool: transform input.txt → output.txt with a
  ``bash_command`` shell pipeline (``cat … | tr … > …``), then read back. Only
  this scenario exposes the real ``bash_command`` tool.

Run recipe (live, against a local model + observability stack)
--------------------------------------------------------------
.. code-block:: sh

    # 1. Start Ollama and pull a tool-capable model.
    ollama serve &              # or run the Ollama app
    ollama pull llama3.2        # default model; passes the #41 capability guard

    # 2. (optional) Start the local observability stack and forward traces.
    #    See observability/ for the compose stack (Tempo + Loki + Grafana).
    export SPORE_OTLP_ENDPOINT=http://localhost:4317

    # 3. Run a scenario. Prompt/model/endpoint/workspace come from args+env.
    uv run --package spore-eval spore-e2e-agent s1 --model llama3.2
    uv run --package spore-eval spore-e2e-agent s2
    uv run --package spore-eval spore-e2e-agent s3
    uv run --package spore-eval spore-e2e-agent s4

    # 4. Verify the grouped trace in Tempo (the run prints the session id):
    curl -s http://localhost:3200/api/traces/<trace_id> | jq '.batches | length'
    #    For S3, spot-check a Compaction span appears mid-trace.

Environment variables (all optional)
* ``SPORE_OLLAMA_MODEL``    — default model id (overridden by ``--model``).
* ``SPORE_OLLAMA_BASE_URL`` — Ollama base url (default http://localhost:11434).
* ``SPORE_OTLP_ENDPOINT``   — when set, forward spans to Tempo (issue #50).
* ``SPORE_E2E_WORKSPACE``   — workspace root (default: a temp dir per run).

Offline / hermetic mode
-----------------------
``--mock`` runs the same scenario builders against a scripted ``MockAgent``,
requiring no Ollama or network. The hermetic CI assertions live in
``tests/test_e2e_scenarios.py``, which drive the same :func:`build_scenario`
path with mock components.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from spore_core.agent import AgentId, ModelAgent
from spore_core.cache_provider import OllamaCacheProvider
from spore_core.context import CompactionConfig
from spore_core.harness import (
    HarnessRunOptions,
    LoopStrategyReAct,
    RunResultSuccess,
    SessionId,
    SessionState,
    Task,
    WorkspaceConfig,
)
from spore_core.memory import now
from spore_core.observability_outbox import OutboxConfig, OutboxObservabilityProvider
from spore_core.ollama import OllamaModelInterface
from spore_core.sandbox import WorkspaceScopedSandbox
from spore_core.storage import InMemoryStorageProvider

from .scenarios import (
    CompleteOnFinalResponse,
    RealToolRegistry,
    ScenarioComponents,
    ScenarioId,
    build_real_tool_registry,
    build_rich_context_manager,
    build_scenario,
    seed_compaction_state,
)


def _arg_value(args: list[str], flag: str) -> str | None:
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            return args[i + 1]
    return None


def _prepare_workspace(scenario: ScenarioId, workspace: Path) -> None:
    """Seed scenario-specific workspace files."""
    if scenario in (ScenarioId.S1, ScenarioId.S5):
        (workspace / "input.txt").write_text("hello from the spore harness end to end scenario\n")


async def _run_scenario(
    scenario: ScenarioId,
    harness: object,
    session_id: SessionId,
    window_limit: int,
) -> object:
    """Drive the scenario, including the S2 multi-turn and S3 seed."""
    strategy = LoopStrategyReAct(max_iterations=8)

    if scenario is ScenarioId.S2:
        task1 = Task.new(scenario.prompt(), session_id, strategy)
        r1 = await harness.run(HarnessRunOptions(task1))  # type: ignore[attr-defined]
        if not isinstance(r1, RunResultSuccess):
            print(f"S2 turn 1 did not succeed: {r1!r}", file=sys.stderr)
            return r1
        # Carry session state across turns. The opaque envelope is empty here in
        # the default ContextManager path; the same SessionId is what ties the
        # two turns together in the trace.
        state = SessionState()
        task2 = Task.new(
            "Add a second TODO item to notes.md that references the first item you "
            "wrote. Use write_file. Reply DONE when finished.",
            session_id,
            strategy,
        )
        return await harness.run(  # type: ignore[attr-defined]
            HarnessRunOptions(task2, session_state=state)
        )

    if scenario is ScenarioId.S3:
        task = Task.new(scenario.prompt(), session_id, strategy)
        state = SessionState()
        seed_compaction_state(
            state,
            "deploy the payment service",
            session_id,
            task.id,
            window_limit,
            int(window_limit * 0.82),
            12,
        )
        return await harness.run(  # type: ignore[attr-defined]
            HarnessRunOptions(task, session_state=state)
        )

    task = Task.new(scenario.prompt(), session_id, strategy)
    return await harness.run(HarnessRunOptions(task))  # type: ignore[attr-defined]


async def _run_live(
    scenario: ScenarioId,
    session_id: SessionId,
    model_id: str,
    workspace: Path,
    obs_root: Path,
) -> object:
    """Run the scenario against a live Ollama model."""
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    agent = ModelAgent(AgentId("e2e-agent"), model)

    registry = build_real_tool_registry(scenario)
    sandbox = WorkspaceScopedSandbox(
        WorkspaceConfig(root=workspace, read_only=False, max_file_size=0)
    )
    bridge = RealToolRegistry(registry, sandbox, session_id, InMemoryStorageProvider())
    tool_schemas = bridge.model_schemas()

    window_limit = 200 if scenario is ScenarioId.S3 else 128_000
    cfg = CompactionConfig(
        threshold=0.80,
        preserve_recent_n=2,
        head_tail_tokens=64,
        offload_path=workspace / ".spore" / "offload",
        max_compaction_attempts=2,
    )
    context_manager = build_rich_context_manager(model, OllamaCacheProvider(), cfg)

    obs = OutboxObservabilityProvider(OutboxConfig(root=obs_root))

    harness = build_scenario(
        scenario,
        ScenarioComponents(
            agent=agent,
            tools=bridge,
            sandbox=sandbox,
            context_manager=context_manager,
            termination_policy=CompleteOnFinalResponse(),
            tool_schemas=tool_schemas,
            observability=obs,
        ),
    )
    return await _run_scenario(scenario, harness, session_id, window_limit)


async def _run_mock(scenario: ScenarioId, session_id: SessionId) -> object:
    """Offline mode: drive the same builder with a scripted mock agent.

    Mirrors the simplest hermetic path; the full per-scenario assertions live in
    ``tests/test_e2e_scenarios.py``.
    """
    from spore_core.agent import MockAgent
    from spore_core.harness import (
        AllowAllSandbox,
        AlwaysContinuePolicy,
        NoopContextManager,
        ScriptedToolRegistry,
        ToolOutputSuccess,
    )
    from spore_core.agent import FinalResponse, ToolCallRequested
    from spore_core.model import TokenUsage, ToolCall

    usage = TokenUsage(input_tokens=10, output_tokens=5)
    agent = MockAgent(AgentId("mock"))
    agent.push(
        ToolCallRequested(
            calls=[ToolCall(id="c1", name="read_file", input={"path": "input.txt"})],
            usage=usage,
        )
    )
    agent.push(FinalResponse(content="DONE", usage=usage))
    tools = ScriptedToolRegistry()
    tools.push(ToolOutputSuccess(content="contents"))

    harness = build_scenario(
        scenario,
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
    task = Task.new(scenario.prompt(), session_id, LoopStrategyReAct(max_iterations=5))
    return await harness.run(HarnessRunOptions(task))


def main() -> None:
    args = sys.argv[1:]
    scenario = ScenarioId.parse(args[0]) if args else None
    if scenario is None:
        print("usage: spore-e2e-agent <s1|s2|s3|s4|s5> [--model <id>] [--mock]", file=sys.stderr)
        raise SystemExit(2)

    mock = "--mock" in args
    model_id = _arg_value(args, "--model") or os.environ.get("SPORE_OLLAMA_MODEL") or "llama3.2"

    safe_ts = now().replace(":", "-").replace(".", "-")
    session_id = SessionId(f"e2e-{scenario.value}-{safe_ts}")

    workspace = Path(
        os.environ.get("SPORE_E2E_WORKSPACE")
        or os.path.join(tempfile.gettempdir(), f"spore-e2e-{session_id}")
    )
    workspace.mkdir(parents=True, exist_ok=True)
    _prepare_workspace(scenario, workspace)
    print(f"workspace  : {workspace}")
    print(f"session_id : {session_id}")

    obs_root = Path(__file__).resolve().parents[4] / ".spore"
    endpoint = os.environ.get("SPORE_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        print(
            "note: SPORE_OTLP_ENDPOINT is unset — writing JSONL only (no Tempo forwarding).",
            file=sys.stderr,
        )

    if mock:
        result = asyncio.run(_run_mock(scenario, session_id))
    else:
        result = asyncio.run(_run_live(scenario, session_id, model_id, workspace, obs_root))

    if isinstance(result, RunResultSuccess):
        print(f"result     : Success ({result.turns} turns)")
        print(f"output     : {result.output!r}")
    else:
        print(f"result     : {result!r}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
