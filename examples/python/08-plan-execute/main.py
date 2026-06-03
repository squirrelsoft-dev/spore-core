"""spore-core example 08 — multi-step goal decomposition with PlanExecute.

This is the first example to swap the **loop strategy**. Everything else — the
``conversational(model)`` builder, the :class:`WorkspaceScopedSandbox`, and the
tool set (``web_search`` + ``write_file`` + ``read_file``, identical to 06) — is
held constant. The ONLY substantive change is one line on the :class:`Task`::

    # 06 — react step-by-step:
    LoopStrategyReAct(max_iterations=10)
    # 08 — decompose the goal first, then execute each subtask:
    LoopStrategyPlanExecute()

With ``PlanExecute``, the harness runs one constrained planner turn FIRST: the
model must return strict JSON ``{"tasks": [...], "rationale": ...}``. That plan is
captured into a :class:`PlanArtifact`, surfaced, then each subtask runs in a
bounded sub-loop. The turn budget is divided across subtasks (per-task cap =
remaining_turns / remaining_tasks), so we set a generous ``max_turns``.

Surfacing the plan — via lifecycle HOOKS, not stream events
-----------------------------------------------------------

There are no plan/subtask *stream* events; the plan is visible through the hook
chain. We register a :class:`PlanExecuteReporter` (a :class:`Hook`) on two events:

- ``OnPlanCreated`` (:class:`OnPlanCreatedContext`) fires post-capture /
  pre-execute — we print a ``── plan ──`` banner: the rationale, then the
  numbered tasks.
- ``OnTaskAdvance`` (:class:`OnTaskAdvanceContext`) fires before each subtask,
  carrying the full ``task`` plus ``task_index`` (0-based) and ``total_tasks`` —
  we print ``[i/N] <instruction>`` with ``i = task_index + 1``.

The plan is also persisted to ``session_state.extras[PLAN_EXECUTE_EXTRAS_KEY]``;
we read it back on success to confirm it round-tripped.

Tools wired (all from the built-in catalogue, identical to 06):

- ``web_search`` — a :class:`WebSearchTool` built via
  :meth:`WebSearchTool.with_config` (#108); issues
  ``GET <SPORE_WEB_SEARCH_ENDPOINT>?q=<query>`` against a SearXNG JSON endpoint.
- ``write_file`` — the agent writes ``async-comparison.md`` into ``workspace/``.
- ``read_file`` — lets the agent re-read what it wrote.

Run it::

    ollama serve &
    ollama pull llama3.2
    export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"  # SearXNG JSON
    uv run main.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from spore_core import (
    PLAN_EXECUTE_EXTRAS_KEY,
    BudgetLimits,
    HarnessBuilder,
    HarnessRunOptions,
    HookContext,
    HookContinue,
    HookDecision,
    HookEvent,
    HookSync,
    LoopStrategyPlanExecute,
    OllamaModelInterface,
    OnPlanCreatedContext,
    OnTaskAdvanceContext,
    RunResultSuccess,
    StandardHookChain,
    StreamToolCall,
    StreamToolResult,
    StreamTurnStart,
    Task,
    WorkspaceConfig,
    WorkspaceScopedSandbox,
    new_session_id,
)
from spore_tools import StandardTool, StandardTools, WebSearchTool
from spore_tools.tools.web import SearchMethod, WebSearchConfig

SYSTEM_PROMPT = (
    "You are a planning research agent. Decompose the goal into clear subtasks. "
    "For each subtask, use web_search to find current information, then synthesize "
    "a clear, cited comparison and save the final document with write_file. Act "
    "using tools — do not answer from memory alone."
)


class PlanExecuteReporter:
    """Lifecycle hook that prints the PlanExecute plan and each subtask as it runs.

    ``OnPlanCreated`` fires once, after the planner turn captures the plan and
    before any subtask executes — the money moment for PlanExecute.
    ``OnTaskAdvance`` fires before each subtask. Both are sync, post/pre,
    plan/task-carrying events. This hook only observes; it always returns
    :class:`HookContinue`. It satisfies the :class:`Hook` Protocol structurally.
    """

    async def handle(self, ctx: HookContext) -> HookDecision:
        if isinstance(ctx, OnPlanCreatedContext):
            print("\n── plan ──")
            if ctx.plan.rationale.strip():
                print(f"rationale: {ctx.plan.rationale}")
            for i, task in enumerate(ctx.plan.tasks):
                print(f"  {i + 1}. {task}")
            print("──────────\n")
        elif isinstance(ctx, OnTaskAdvanceContext):
            print(f"[{ctx.task_index + 1}/{ctx.total_tasks}] {ctx.task.instruction}")
        return HookContinue()

    def events(self) -> list[HookEvent]:
        return [HookEvent.ON_PLAN_CREATED, HookEvent.ON_TASK_ADVANCE]

    def name(self) -> str:
        return "plan-execute-reporter"

    def sync_mode(self) -> HookSync:
        return HookSync.SYNC


def _truncate(text: str, limit: int = 200) -> str:
    """Keep observe lines readable — search results can be long."""
    flat = text.replace("\n", " ")
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


async def main() -> int:
    parser = argparse.ArgumentParser(description="spore-core plan-execute agent")
    parser.add_argument("--model")
    parser.add_argument("--prompt")
    args = parser.parse_args()

    model_id = args.model or os.environ.get("SPORE_OLLAMA_MODEL") or "llama3.2"
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)

    # The search backend endpoint. ``web_search`` issues
    # ``GET <endpoint>?q=<query>`` and returns the JSON body to the agent. There
    # is no live backend in spore-core, so you must supply one — a self-hosted
    # SearXNG JSON endpoint (``.../search?format=json``). #108 added
    # ``WebSearchConfig`` so the GET method + query param are configurable; the
    # GET path preserves the ``?format=json`` already on the endpoint.
    endpoint = (os.environ.get("SPORE_WEB_SEARCH_ENDPOINT") or "").strip()
    if not endpoint:
        print(
            "SPORE_WEB_SEARCH_ENDPOINT is not set.\n"
            "Set it to a SearXNG JSON search endpoint, e.g. "
            "http://localhost:8888/search?format=json\n"
            "See .env.example and the README.",
            file=sys.stderr,
        )
        return 2

    # The agent operates inside this example's ``workspace/`` directory. Resolve
    # it relative to this source file so ``uv run main.py`` works from anywhere,
    # and canonicalize it — the sandbox requires a canonical, existing root.
    workspace_root = Path(__file__).parent / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace_root = workspace_root.resolve(strict=True)

    prompt = args.prompt or (
        # A multi-step goal that benefits from upfront decomposition: search each
        # runtime, synthesize a comparison, then write the file.
        "Research the Rust async ecosystem, write a comparison of tokio vs async-std "
        "vs smol covering performance, ecosystem maturity, and use cases, and save it "
        "to async-comparison.md."
    )

    # Register the plan reporter on a StandardHookChain. The chain is how the plan
    # becomes visible: there are no plan/subtask stream events.
    chain = StandardHookChain()
    chain.register(PlanExecuteReporter())

    # Same ``conversational`` harness + ``WorkspaceScopedSandbox`` + tool set as
    # 06. The ONLY substantive change vs 06 is the ``LoopStrategy`` on the Task
    # below.
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    sandbox = WorkspaceScopedSandbox(WorkspaceConfig(root=workspace_root))
    # SearXNG's JSON API is ``GET /search?q=<query>&format=json``. Configure the
    # tool for GET with the query keyed under ``q``; no auth is needed.
    web_search_config = WebSearchConfig(
        endpoint=endpoint,
        method=SearchMethod.GET,
        query_param="q",
        auth_headers=[],
        body_auth_params=[],
    )
    web_search = StandardTool(WebSearchTool.with_config(web_search_config), WebSearchTool.schema())
    harness = (
        HarnessBuilder.conversational(model)
        .sandbox(sandbox)
        .tool(web_search)
        .tool(StandardTools.write_file())
        .tool(StandardTools.read_file())
        .system_prompt(SYSTEM_PROMPT)
        .hooks(chain)
        .build()
    )

    # THE ONE-LINE SWAP. 06 used ``LoopStrategyReAct(max_iterations=10)``; here we
    # decompose first via PlanExecute. The turn budget is divided across subtasks
    # (per-task cap = remaining_turns / remaining_tasks), so we give it generous
    # headroom — an 8-step plan at 64 turns gives each subtask ~8 instead of starving.
    task = Task.new(
        prompt,
        new_session_id(),
        LoopStrategyPlanExecute(),
        budget=BudgetLimits(max_turns=64),
    )

    # Print each turn (Think) and each tool call + result (Act / Observe). This is
    # most useful for the plan-phase turn; the Python harness suppresses the
    # subtask sub-loop stream (like Rust/Go), so the hooks above are the portable
    # view of execution.
    def on_stream(event: object) -> None:
        if isinstance(event, StreamTurnStart):
            print(f"think  · turn {event.turn}")
        elif isinstance(event, StreamToolCall):
            print(f"    act    → {event.name}({json.dumps(event.args)})")
        elif isinstance(event, StreamToolResult):
            tag = "obs(err)" if event.is_error else "obs "
            print(f"    {tag}→ {_truncate(event.content)}")

    options = HarnessRunOptions(task, on_stream=on_stream)

    print(f"model    : {model_id}")
    print(f"endpoint : {endpoint}")
    print(f"workspace: {workspace_root}")
    print("strategy : PlanExecute (06 used ReAct)")
    print(f"prompt   : {prompt}\n")

    try:
        result = await harness.run(options)
    except OSError as e:
        # Ollama unreachable / endpoint refused the connection, etc.
        print(f"\ncould not reach the model — is `ollama serve` running? ({e})", file=sys.stderr)
        return 1

    if isinstance(result, RunResultSuccess):
        print(f"\nanswer ({result.turns} turn(s)): {result.output}")
        # The captured plan is persisted in extras — confirm it round-tripped.
        plan = result.session_state.extras.get(PLAN_EXECUTE_EXTRAS_KEY)
        if isinstance(plan, dict) and isinstance(plan.get("tasks"), list):
            print(
                f'\nplan persisted in extras["{PLAN_EXECUTE_EXTRAS_KEY}"] '
                f"with {len(plan['tasks'])} subtask(s)"
            )
        doc = workspace_root / "async-comparison.md"
        if doc.exists():
            print(f"\nasync-comparison.md now exists on disk: {doc}")
        return 0

    print(f"\nrun did not succeed: {result!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
