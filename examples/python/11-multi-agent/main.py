"""spore-core example 11 — multi-agent composition: agents as tools.

The thesis: agents are composable
---------------------------------

**The harness does not care whether a "tool" dispatches to a function or to
*another agent*.** This example builds three agents and wires two of them into
the third as ordinary tools.

The three agents
----------------

- **research worker** — a :class:`Harness` with exactly one tool, ``web_search``
  (the SearXNG-backed :class:`WebSearchTool` from example 06). Given an
  instruction string, it searches the web and returns raw, cited findings as its
  output string.
- **writing worker** — a :class:`Harness` with NO tools. Given the research
  findings as its instruction, it formats them into a polished markdown report
  and returns that markdown as its output string. It never touches the network —
  it only shapes prose.
- **orchestrator** — a :class:`Harness` whose "tools" are the two workers
  (wrapped as :class:`SubagentTool` s) plus ``write_file``. It plans the job,
  calls ``research_worker``, hands that output to ``writing_worker``, then writes
  the final markdown to ``workspace/report.md``.

Three agents, two handoffs (``research → writing``, ``writing → report.md``),
one output.

The agent-as-tool mechanism
---------------------------

Each worker is a fully-built child :class:`Harness` wrapped in a
:class:`SubagentTool`. ``SubagentTool`` implements the same ``Tool`` protocol as
``write_file`` or ``web_search``: when the orchestrator emits a tool call for
``research_worker``, the tool reads a single ``instruction`` string from the
call, runs the child harness on a fresh :class:`Task`, and returns the child's
final output string as the tool result. The orchestrator cannot tell — and does
not need to know — that the "tool" behind ``research_worker`` is an entire agent
with its own loop, its own model, and its own ``web_search`` tool.

We register each worker on the orchestrator's builder the same way example 06
registers ``web_search``: build a :class:`StandardTool` from the
``SubagentTool`` plus a :class:`ToolSchema` advertising the
``{ "instruction": string }`` input, then ``.tool(...)`` it.

Why this keeps the orchestrator's context clean
-----------------------------------------------

Both workers use :class:`ContextSharingIsolated`: each runs in a brand-new
session with NO shared mutable state with the orchestrator or with each other.
The research worker may burn a dozen internal ReAct turns issuing search queries
and sifting noisy JSON — but the ONLY thing that crosses back into the
orchestrator's context is the worker's final output string. The orchestrator
never sees the worker's intermediate turns, failed searches, or raw result
blobs. The noisy work is encapsulated; the orchestrator's context stays small
and on-topic. This is the whole reason to delegate to a subagent rather than
inline the work.

A direct, visible consequence: the child's internal turns do **not** stream up
through the parent. The orchestrator's ``on_stream`` callback only sees the
``StreamToolCall`` to ``research_worker`` and the ``StreamToolResult`` coming
back — which is exactly the agent boundary we print. The invisibility of the
child's turns is not a limitation; it *is* the context isolation, made
observable.

The strategy split: PlanExecute at the top, ReAct inside
--------------------------------------------------------

The orchestrator runs :class:`LoopStrategyPlanExecute`: it decomposes the job
("research, then write, then save") into subtasks up front and executes them in
order — natural for a coordinator. Each worker, by contrast, runs ReAct
internally. (The ReAct loop is hardcoded inside ``SubagentTool``; a subagent
always runs its child as ``LoopStrategyReAct``.) So the two layers use two
different loop strategies, each fit to its level: deliberate planning at the
orchestrator, step-by-step tool use inside the workers.

Reading the output: agent boundaries
-------------------------------------

The point of this example is *legibility*: you should be able to read stdout and
see which agent is acting, what it received, and what it returned. The
orchestrator's stream fires a ``StreamToolCall(name, args)`` and a
``StreamToolResult`` for each worker dispatch — we turn those into a boxed
banner::

    ┌─ orchestrator → research_worker
    │  received: <instruction>
    └─ research_worker → orchestrator
       returned: <truncated findings>

The plan itself surfaces through a lifecycle HOOK, not a stream event:
PlanExecute has no plan/subtask stream events, so we register a
:class:`PlanReporter` on ``OnPlanCreated`` / ``OnTaskAdvance`` (same pattern as
example 08).

``StreamToolResult`` carries only ``call_id`` (no tool name), so we remember
which ``call_id`` belonged to which tool when the ``StreamToolCall`` fires, then
look it up on the result to label the closing half of each boundary.

Run it::

    ollama serve &
    ollama pull llama3.2
    export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"  # SearXNG JSON
    uv run main.py
    uv run main.py --topic "the history of the TCP/IP protocol suite"
    uv run main.py --model qwen2.5-coder:7b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from spore_core import (
    BudgetLimits,
    Harness,
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
    RegistryToolSchema,
    RunResultSuccess,
    StandardHookChain,
    StandardToolRegistry,
    StreamToolCall,
    StreamToolResult,
    StreamTurnStart,
    Task,
    ToolAnnotations,
    WorkspaceConfig,
    WorkspaceScopedSandbox,
    new_session_id,
)
from spore_tools import StandardTool, StandardTools, WebSearchTool
from spore_tools.tools.subagent import ContextSharingIsolated, SubagentTool
from spore_tools.tools.web import SearchMethod, WebSearchConfig

# Per-worker wall-clock cap (seconds). A worker can burn many internal ReAct
# turns; this bounds how long the orchestrator will wait on any single
# delegation. ``SubagentTool`` enforces it via ``asyncio.wait_for``.
WORKER_TIMEOUT_SECONDS = 180.0

# Names of the two worker agents (vs. the plain ``write_file`` tool). Used to
# decide which stream events get the boxed agent-boundary banner.
RESEARCH_WORKER = "research_worker"
WRITING_WORKER = "writing_worker"

RESEARCH_PROMPT = (
    "You are a research worker. Use the web_search tool to gather current, factual "
    "information on the topic you are given. Issue focused queries, read the results, "
    "and return a concise set of findings as plain text — key facts, figures, and "
    "definitions — each followed by the source URL it came from. Do NOT format a "
    "report; just return the raw, cited findings. Act using web_search — do not "
    "answer from memory alone."
)

WRITING_PROMPT = (
    "You are a writing worker. You will be given a set of raw, cited research "
    "findings. Turn them into a polished markdown report: a top-level `# ` title, a "
    "short intro, well-organized `## ` sections, and a `## Sources` list preserving "
    "the URLs from the findings. Return ONLY the markdown of the report — no "
    "preamble, no commentary. You have no tools; produce the report directly as your "
    "final answer."
)

ORCHESTRATOR_PROMPT = (
    "You are an orchestrator. You coordinate two worker agents, each exposed to you "
    "as a tool. Your plan is always the same three steps: (1) call research_worker "
    "with an `instruction` describing the topic to research; (2) call writing_worker "
    "with an `instruction` that is the EXACT findings text returned by the research "
    "worker, asking it to format a polished markdown report; (3) call write_file to "
    "save the writing worker's markdown verbatim to `report.md`. Do the research and "
    "writing by delegating to the workers — never do it yourself — and always finish "
    "by writing report.md."
)


def _instruction_schema() -> dict[str, Any]:
    """The single-parameter input schema every worker tool advertises: the
    orchestrator passes one ``instruction`` string, which ``SubagentTool``
    forwards to the child harness as its task. Matches the schema
    ``SubagentTool`` reads on execute."""
    return {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "The full instruction / task for the worker agent.",
            }
        },
        "required": ["instruction"],
    }


def _build_web_search(endpoint: str) -> StandardTool:
    """Build the SearXNG-backed ``web_search`` catalogue tool (identical wiring to
    example 06). Only the research worker gets this. SearXNG's JSON API is
    ``GET /search?q=<query>&format=json``: configure GET with the query keyed
    under ``q``; no auth is needed."""
    config = WebSearchConfig(
        endpoint=endpoint,
        method=SearchMethod.GET,
        query_param="q",
        auth_headers=[],
        body_auth_params=[],
    )
    return StandardTool(WebSearchTool.with_config(config), WebSearchTool.schema())


def _build_research_harness(model_id: str, base_url: str, endpoint: str) -> Harness:
    """Build the research worker: a child harness whose only tool is
    ``web_search``. Each agent gets its OWN fresh model instance — the workers
    are genuinely independent and do not share a model object with the
    orchestrator."""
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    return (
        HarnessBuilder.conversational(model)
        .tool(_build_web_search(endpoint))
        .system_prompt(RESEARCH_PROMPT)
        .build()
    )


def _build_writing_harness(model_id: str, base_url: str) -> Harness:
    """Build the writing worker: a child harness with NO tools — it formats prose
    and returns the report as its final answer."""
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    return HarnessBuilder.conversational(model).system_prompt(WRITING_PROMPT).build()


def _build_worker_tool(name: str, description: str, child: Harness) -> StandardTool:
    """Wrap a child harness as a ``SubagentTool`` and bundle it into a
    ``StandardTool`` the orchestrator can register — exactly how example 06 wraps
    ``web_search``, only the "implementation" here is an entire agent."""
    # ``child_registry`` is used ONLY for the depth-1 ``has_subagent_tools()``
    # check. The workers have no subagent tools of their own, so a fresh empty
    # registry passes trivially. The child's REAL tools were wired on its builder.
    empty_child_registry = StandardToolRegistry()
    subagent = SubagentTool.new(
        name=name,
        description=description,
        input_schema=_instruction_schema(),
        timeout_seconds=WORKER_TIMEOUT_SECONDS,
        context_sharing=ContextSharingIsolated(),
        harness=child,
        child_registry=empty_child_registry,
    )
    schema = RegistryToolSchema(
        name=name,
        description=description,
        parameters=_instruction_schema(),
        # ``open_world``: a subagent reaches outside the process (it runs a whole
        # agent, and the research worker hits the network), so it is not a closed,
        # read-only computation.
        annotations=ToolAnnotations(open_world=True),
    )
    return StandardTool(subagent, schema)


class PlanReporter:
    """Lifecycle hook that prints the orchestrator's PlanExecute plan and each
    subtask as it advances. PlanExecute has NO plan/subtask stream events, so the
    hook chain is how the plan becomes visible (same pattern as example 08).

    Satisfies the ``Hook`` Protocol structurally; it only observes and always
    returns :class:`HookContinue`.
    """

    async def handle(self, ctx: HookContext) -> HookDecision:
        if isinstance(ctx, OnPlanCreatedContext):
            print("\n── orchestrator plan ──")
            if ctx.plan.rationale.strip():
                print(f"rationale: {ctx.plan.rationale}")
            for i, task in enumerate(ctx.plan.tasks):
                print(f"  {i + 1}. {task}")
            print("───────────────────────\n")
        elif isinstance(ctx, OnTaskAdvanceContext):
            print(f"[{ctx.task_index + 1}/{ctx.total_tasks}] {ctx.task.instruction}")
        return HookContinue()

    def events(self) -> list[HookEvent]:
        return [HookEvent.ON_PLAN_CREATED, HookEvent.ON_TASK_ADVANCE]

    def name(self) -> str:
        return "orchestrator-plan-reporter"

    def sync_mode(self) -> HookSync:
        return HookSync.SYNC


def _is_worker(name: str) -> bool:
    """A tool name that maps to one of the two worker agents (vs. ``write_file``)."""
    return name in (RESEARCH_WORKER, WRITING_WORKER)


def _truncate(text: str, limit: int = 280) -> str:
    """Keep boundary lines readable — findings and reports can be long."""
    flat = text.replace("\n", " ")
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


async def main() -> int:
    parser = argparse.ArgumentParser(description="spore-core multi-agent orchestrator")
    parser.add_argument("--model")
    parser.add_argument("--topic")
    args = parser.parse_args()

    model_id = args.model or os.environ.get("SPORE_OLLAMA_MODEL") or "llama3.2"
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)

    # Same required search backend as example 06 — only the research worker uses
    # it, but the orchestrator cannot do its job without it, so we fail fast here
    # with the same message and exit code (2) example 06 uses.
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

    # The orchestrator operates inside this example's ``workspace/`` directory; it
    # writes the final report there. Resolve it relative to this source file so
    # ``uv run main.py`` works from anywhere, and canonicalize it — the sandbox
    # requires a canonical, existing root.
    workspace_root = Path(__file__).parent / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace_root = workspace_root.resolve(strict=True)

    topic = args.topic or (
        # A TIMELESS, encyclopedic subject so web-search results stay stable and
        # useful across runs (per the issue: keep the topic generic). Matches the
        # Rust reference so all four language examples research the same thing.
        "the history and core ideas of the Rust programming language"
    )
    prompt = (
        f"Research {topic} and produce a polished markdown report saved to report.md. "
        "Delegate the research to research_worker and the writing to writing_worker."
    )

    # ---- Build the two workers, then wrap them as orchestrator tools --------
    research_child = _build_research_harness(model_id, base_url, endpoint)
    writing_child = _build_writing_harness(model_id, base_url)

    research_tool = _build_worker_tool(
        RESEARCH_WORKER,
        "Delegate to the research agent: pass an `instruction` describing a topic; it "
        "web-searches and returns concise, cited findings as text.",
        research_child,
    )
    writing_tool = _build_worker_tool(
        WRITING_WORKER,
        "Delegate to the writing agent: pass an `instruction` containing research "
        "findings; it returns a polished markdown report.",
        writing_child,
    )

    # ---- Build the orchestrator: workers-as-tools + write_file --------------
    # Each agent gets its own fresh model instance; the orchestrator's is built
    # here, the workers' inside their build functions above.
    orchestrator_model = OllamaModelInterface.with_base_url(model_id, base_url)
    sandbox = WorkspaceScopedSandbox(WorkspaceConfig(root=workspace_root))

    # The plan surfaces through a hook (PlanExecute has no plan stream events).
    chain = StandardHookChain()
    chain.register(PlanReporter())

    orchestrator = (
        HarnessBuilder.conversational(orchestrator_model)
        .sandbox(sandbox)
        .tool(research_tool)
        .tool(writing_tool)
        .tool(StandardTools.write_file())
        .system_prompt(ORCHESTRATOR_PROMPT)
        .hooks(chain)
        .build()
    )

    # The orchestrator plans the three steps up front via PlanExecute, then
    # executes them. The turn budget is divided across subtasks, so give it
    # generous headroom — each worker dispatch may itself be slow.
    task = Task.new(
        prompt,
        new_session_id(),
        LoopStrategyPlanExecute(),
        budget=BudgetLimits(max_turns=32),
    )

    # The orchestrator's stream is where the agent boundaries become visible. A
    # ``StreamToolCall`` to a worker IS the "→ worker" boundary; the matching
    # ``StreamToolResult`` IS the "← worker" boundary. The child's own internal
    # turns do NOT appear here — that invisibility is the context isolation, made
    # observable (see the module docstring).
    #
    # ``StreamToolResult`` carries only ``call_id`` (no tool name), so we remember
    # which ``call_id`` belonged to which tool when the ``StreamToolCall`` fires,
    # then look it up on the result to label the closing half of the boundary.
    call_names: dict[str, str] = {}

    def on_stream(event: object) -> None:
        if isinstance(event, StreamTurnStart):
            print(f"orchestrator · plan/execute turn {event.turn}")
        elif isinstance(event, StreamToolCall):
            call_names[event.call_id] = event.name
            if _is_worker(event.name):
                instruction = event.args.get("instruction", "<no instruction>")
                print(f"┌─ orchestrator → {event.name}")
                print(f"│  received: {_truncate(str(instruction), 200)}")
            else:
                print(f"  orchestrator → {event.name}({_truncate(json.dumps(event.args), 160)})")
        elif isinstance(event, StreamToolResult):
            name = call_names.pop(event.call_id, "<tool>")
            if _is_worker(name):
                tag = "FAILED" if event.is_error else "returned"
                print(f"└─ {name} → orchestrator")
                print(f"   {tag}: {_truncate(event.content, 280)}")
            else:
                tag = "err" if event.is_error else "ok"
                print(f"  {name} → orchestrator [{tag}]: {_truncate(event.content, 160)}")

    options = HarnessRunOptions(task, on_stream=on_stream)

    print(f"model       : {model_id}")
    print(f"endpoint    : {endpoint}")
    print(f"workspace   : {workspace_root}")
    print("strategy    : orchestrator=PlanExecute, workers=ReAct (isolated)")
    print(f"agents      : orchestrator → [{RESEARCH_WORKER}, {WRITING_WORKER}]")
    print(f"topic       : {topic}\n")

    try:
        result = await orchestrator.run(options)
    except OSError as e:
        # Ollama unreachable / endpoint refused the connection, etc.
        print(f"\ncould not reach the model — is `ollama serve` running? ({e})", file=sys.stderr)
        return 1

    report = workspace_root / "report.md"
    if isinstance(result, RunResultSuccess):
        print(f"\norchestrator done ({result.turns} turn(s)): {_truncate(result.output, 280)}")
        if report.exists():
            print(f"\nreport.md now exists on disk: {report}")
        else:
            print(
                "\nwarning: orchestrator finished but report.md was not written.", file=sys.stderr
            )
        return 0

    print(f"\nrun did not succeed: {result!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
