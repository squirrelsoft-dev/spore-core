"""spore-core example 08 — multi-step goal decomposition with PlanExecute.

This is the first example to swap the **loop strategy**. Everything else — the
``conversational(model)`` builder, the :class:`WorkspaceScopedSandbox`, and the
tool set (``web_search`` + ``write_file`` + ``read_file``, identical to 06) — is
held constant. The ONLY substantive change is one line on the :class:`Task`::

    # 06 — react step-by-step:
    ReactConfig.per_loop(10)
    # 08 — decompose the goal first, then execute each subtask:
    PlanExecuteConfig(plan=ReactConfig(output="plan-schema", ...), ...)

Post-#119 the ``LoopStrategy`` is a recursive union of config newtypes, so the
strategy is a composed tree rather than a flat literal. ``PlanExecute``'s
``plan`` slot is STRUCTURED — a bare ``ReactConfig`` there MUST declare an
``output`` schema handle (here ``plan-schema``), which
:meth:`ExecutionRegistry.validate` enforces at run entry. The empty agent /
toolset handles on the leaves are default-filled by the builder; only the
``plan-schema`` handle is registered explicitly on the
:class:`ExecutionRegistry` wired via ``.registry(...)``.

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

Tool-calling mode: this example uses **native Ollama tool calling by default**
(the real typed tool schema), which works for tool-capable / cloud models like
``gemma4:31b-cloud``. Pass ``--structured`` to opt into
``ModelParams(structured_tool_calls=True)`` — schema-constrained decoding that
helps small local models (e.g. ``llama3.2``) emit one clean tool call per turn
across both the plan and execute phases. Structured mode exposes an
always-available ``final`` envelope, so a capable model may emit
``{"tool":"final"}`` prematurely and return an EMPTY answer; if you see that
(and no ``async-comparison.md``), drop ``--structured``.

Staying within small (~128K) context windows
---------------------------------------------

Under ``PlanExecute``, verbose tool output is retained across every plan step,
so a few searches can overflow a model with a ~128K window (e.g. ``gemma4:e4b``,
131072 tokens). Two measures keep this example running cleanly on such models:

- It **distills ``web_search`` results**: a :class:`ConciseWebSearch` wrapper
  trims the verbatim SearXNG JSON (25-40K tokens/call) down to the top 6 results
  with only ``title`` / ``url`` / ``content``, so context stays small.
- It **lowers the compaction threshold** to ``0.45`` (compaction at ~90K tokens
  instead of the default ~160K), installed via ``.context_manager(...)``, so
  compaction fires before a 128K-window model overflows.

Run it::

    ollama serve &
    ollama pull llama3.2
    export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"  # SearXNG JSON
    uv run main.py                # native tool calling (default)
    uv run main.py --structured   # constrained decoding for small local models
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
    CompactionConfig,
    ExecutionRegistry,
    HarnessBuilder,
    HarnessRunOptions,
    HookContext,
    HookContinue,
    HookDecision,
    HookEvent,
    HookSync,
    LoopStrategy,
    ModelParams,
    NullCacheProvider,
    OllamaModelInterface,
    OnPlanCreatedContext,
    OnTaskAdvanceContext,
    PlanExecuteConfig,
    ReactConfig,
    RunResultSuccess,
    SandboxProvider,
    StandardContextManager,
    StandardHookChain,
    StreamToolCall,
    StreamToolResult,
    StreamTurnStart,
    Task,
    ToolCall,
    ToolContext,
    ToolOutput,
    ToolOutputSuccess,
    WorkspaceConfig,
    WorkspaceScopedSandbox,
    into_harness_adapter,
    new_session_id,
)
from spore_tools import StandardTool, StandardTools, WebSearchTool
from spore_tools.tools.web import SearchMethod, WebSearchConfig

# The GLOBAL operating prompt — the shared capability contract. It is the
# DEFAULT every leaf falls back to. Under SC-10 (#161) the plan and execute
# leaves below each carry their OWN ``system_prompt``, which REPLACES this for
# those phases (each phase sees ONLY its own prompt). This global remains the
# prompt any leaf WITHOUT an override would use.
SYSTEM_PROMPT = (
    "You are a research-and-writing agent. Your ONLY capabilities are: web_search "
    "(find current information online), read_file, and write_file (save your work to "
    "the workspace). You have NO shell or terminal — you cannot install software, set "
    "up projects or environments, run/compile/build code, or execute commands. Act "
    "using tools — do not answer from memory alone."
)

# SC-10 (#161): the PLAN phase's own system prompt. The planner only DECOMPOSES —
# it never executes a subtask, so its prompt is about producing a good plan, not
# about searching/writing. This replaces SYSTEM_PROMPT for the plan leaf only.
PLAN_SYSTEM_PROMPT = (
    "You are the PLANNER. Your ONLY job is to decompose the goal into an ordered "
    "list of subtasks. Each subtask must be achievable with web_search and "
    "write_file alone — there is NO shell or terminal, so never plan setup, "
    "installation, or build steps. Do not perform any subtask yourself; output ONLY "
    "the plan."
)

# SC-10 (#161): the EXECUTE phase's own system prompt. The executor works ONE
# subtask at a time — it does not re-plan. This replaces SYSTEM_PROMPT for the
# execute leaf only, so plan-phase decomposition guidance never leaks into
# execution.
EXECUTE_SYSTEM_PROMPT = (
    "You are the EXECUTOR. You are given ONE subtask at a time. Use web_search to "
    "gather current information for it, then synthesize a clear, cited result and "
    "save your work with write_file. Do not re-plan or invent new subtasks — "
    "complete the one you were given, using tools rather than memory."
)


def plan_schema() -> dict[str, object]:
    """The plan-phase output contract (``plan-schema``). Post-#119,
    ``PlanExecute``'s ``plan`` slot is STRUCTURED: a bare ``ReactConfig`` there
    must declare an ``output`` schema so the slot yields a typed task graph
    (:meth:`ExecutionRegistry.validate` enforces this via the structured-slot
    check). This is the strict JSON the planner turn returns."""
    return {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "Ordered subtasks to execute in sequence.",
                "items": {"type": "string"},
            },
            "rationale": {"type": "string"},
        },
        "required": ["tasks"],
    }


def build_registry() -> ExecutionRegistry:
    """The registry the composed strategy's handles resolve against. The single
    ``plan-schema`` slot is the only EXPLICIT handle; the builder default-fills the
    empty agent/toolset handles (``ReactConfig.per_loop`` uses empty handles) from
    the harness's own model and global tool catalogue at ``build``."""
    return ExecutionRegistry.builder().schema("plan-schema", plan_schema()).build()


def plan_execute_strategy() -> LoopStrategy:
    """The post-#119 composed strategy: ``PlanExecute(plan: ReAct, execute: ReAct)``.
    The plan leaf carries the ``plan-schema`` output contract (required for the
    structured ``plan`` slot); both leaves use empty agent/toolset handles that the
    builder default-fills. Old flat shape was ``LoopStrategyPlanExecute()``. The
    plan leaf is bare and effectively unbounded (``per_loop(2**31 - 1)``); the
    global ``max_turns`` backstop bounds the whole run.

    SC-10 (#161, per-leaf system prompt): the plan and execute leaves each carry
    their OWN ``system_prompt``. The plan phase runs under ``PLAN_SYSTEM_PROMPT``
    (decompose only) and the execute phase under ``EXECUTE_SYSTEM_PROMPT`` (do one
    subtask) — each phase sees ONLY its own prompt, so planning guidance never
    leaks into execution and vice versa. The global ``SYSTEM_PROMPT`` remains the
    documented fallback for any leaf WITHOUT an override. (The per-leaf TOOLSET
    override is the existing ``ReactConfig.toolset`` handle; here both phases share
    the global catalogue.)"""
    # The plan leaf is a bare ReAct (empty agent/toolset, default-filled by the
    # builder) carrying the structured-slot ``plan-schema`` output contract and an
    # effectively-unbounded ``per_loop`` budget. ``PlanExecuteConfig.simple()``
    # gives both phases bare ReAct leaves; we override the plan leaf's output and
    # wire each leaf's per-phase ``system_prompt`` (SC-10).
    plan = ReactConfig.per_loop(2**31 - 1)
    plan.output = "plan-schema"
    plan.system_prompt = PLAN_SYSTEM_PROMPT
    execute = ReactConfig.per_loop(2**31 - 1)
    execute.system_prompt = EXECUTE_SYSTEM_PROMPT
    strategy = PlanExecuteConfig.simple()
    strategy.plan = plan
    strategy.execute = execute
    return strategy


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


def _distill_search_results(content: str) -> str:
    """Keep only the top 6 results, and for each only ``title`` / ``url`` /
    ``content`` (content clipped to ~500 chars). Drop all other fields and
    top-level keys (``answers``, ``infoboxes``, ``suggestions``,
    ``unresponsive_engines``, …), and re-serialize as compact
    ``{"results": [...]}``.

    Defensive: if the body is not JSON or has no ``results`` array, the original
    string is returned unchanged — we never error just because the shape was
    unexpected.
    """
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return content
    if not isinstance(value, dict):
        return content
    results = value.get("results")
    if not isinstance(results, list):
        return content

    distilled = []
    for r in results[:6]:
        if not isinstance(r, dict):
            continue
        title = r.get("title") if isinstance(r.get("title"), str) else ""
        url = r.get("url") if isinstance(r.get("url"), str) else ""
        body = r.get("content") if isinstance(r.get("content"), str) else ""
        distilled.append({"title": title, "url": url, "content": body[:500]})

    return json.dumps({"results": distilled}, separators=(",", ":"))


class ConciseWebSearch:
    """A thin wrapper around the built-in :class:`WebSearchTool` that distills
    its output.

    WHY: the core ``web_search`` tool returns the SearXNG JSON body VERBATIM (by
    frozen spec — normalization is out of scope for the core tool). Each search
    yields ~25-30 results, each carrying the full ``content`` plus a dozen noise
    fields (``thumbnail``, ``engine``, ``score``, ``parsed_url``, …) — roughly
    25-40K tokens per call. Under ``PlanExecute`` those dumps are retained across
    every plan step, so three searches alone can overflow a ~128K-window model.
    This wrapper keeps only the top results and the fields the agent actually
    reads, so the conversation context stays small. The model still sees an
    identical ``web_search`` tool (same name + schema); only the *result* is
    trimmed. Structural :class:`~spore_core.tool_registry.Tool` impl.
    """

    def __init__(self, inner: WebSearchTool) -> None:
        self._inner = inner

    def name(self) -> str:
        return "web_search"

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return True

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        out = await self._inner.execute(call, sandbox, ctx)
        # Errors and every non-Success variant pass through untouched.
        if isinstance(out, ToolOutputSuccess):
            return ToolOutputSuccess(
                content=_distill_search_results(out.content),
                truncated=out.truncated,
            )
        return out


async def main() -> int:
    parser = argparse.ArgumentParser(description="spore-core plan-execute agent")
    parser.add_argument("--model")
    parser.add_argument("--prompt")
    parser.add_argument(
        "--structured",
        action="store_true",
        help=(
            "Opt into schema-constrained (structured) tool calls for small local "
            "models. Default is native Ollama tool calling, which works for "
            "tool-capable / cloud models like gemma4:31b-cloud."
        ),
    )
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
    # Wrap the raw ``WebSearchTool`` in ``ConciseWebSearch`` so verbose SearXNG
    # JSON is distilled before it enters the conversation (see the class doc
    # above). Same name + schema, so the model sees an identical ``web_search``
    # tool.
    web_search = StandardTool(
        ConciseWebSearch(WebSearchTool.with_config(web_search_config)),
        WebSearchTool.schema(),
    )

    # Lower the compaction threshold so it fires at ~0.45 * 200K ≈ 90K tokens,
    # BEFORE a ~128K-window model (e.g. gemma4:e4b) overflows. ``conversational``
    # installs a ``StandardContextManager`` with ``CompactionConfig()`` defaults
    # (compaction at 80% of a 200K window ≈ 160K), which is too late here. The
    # context manager gets its OWN model instance for summarization turns.
    context_manager = into_harness_adapter(
        StandardContextManager(
            OllamaModelInterface.with_base_url(model_id, base_url),
            NullCacheProvider(),
            CompactionConfig(threshold=0.45),
        )
    )

    harness = (
        HarnessBuilder.conversational(model)
        .sandbox(sandbox)
        .registry(build_registry())
        .context_manager(context_manager)
        .tool(web_search)
        .tool(StandardTools.write_file())
        .tool(StandardTools.read_file())
        .system_prompt(SYSTEM_PROMPT)
        # Native tool calling by default; ``--structured`` opts into constrained
        # decoding for small local models. With structured mode the "think" line
        # is just a turn marker (one clean tool call per turn, no interleaved
        # reasoning), but a capable model can bail early via the always-available
        # ``final`` envelope — see the docstring.
        .model_params(ModelParams(structured_tool_calls=args.structured))
        .hooks(chain)
        .build()
    )

    # THE STRATEGY SWAP. 06 used a bare ``ReactConfig.per_loop(10)``; here we
    # decompose first via the composed ``PlanExecute(plan: ReAct{plan-schema},
    # execute: ReAct)`` tree (post-#119). The plan leaf carries the ``plan-schema``
    # output contract — ``PlanExecute``'s ``plan`` slot is STRUCTURED, so a bare
    # ``ReAct`` there MUST declare an output schema. The turn budget is divided
    # across subtasks (per-task cap = remaining_turns / remaining_tasks), so we give
    # it generous headroom — an 8-step plan at 64 turns gives each subtask ~8
    # instead of starving.
    task = Task.new(
        prompt,
        new_session_id(),
        plan_execute_strategy(),
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
