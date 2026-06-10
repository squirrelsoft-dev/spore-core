"""spore-core example 12 — cordyceps: the capstone of the Composable Execution
refactor (#117–#131).

The thesis: you describe a strategy as DATA — a composed ``LoopStrategy`` tree —
wire its string handles to concrete collaborators in an ``ExecutionRegistry``,
and the harness runs the whole nested machine under one shared budget / usage /
observability context.

The motivating composition is::

    Ralph[ PlanExecute[ ReAct, SelfVerifying[ ReAct ] ] ]
    │       │             │      │             │
    │       │             │      │             └─ worker: audits ONE module
    │       │             │      └─ Default-FAIL evaluator (single read-only turn)
    │       │             └─ plan: explores the repo, builds a blocker-aware DAG
    │       └─ plan→ready-set: walks the DAG in dependency order, self-verifying
    └─ continuation wrapper: resets the window, resumes from durable progress

What changed vs. the pre-#131 example (HONEST note)
---------------------------------------------------

The old depth-1 example used a hand-built ``SubagentTool`` orchestrator with a
per-node consult mediator (#114) and an architect-side ``load_skill`` tool
(#115). The declarative tree has NO SubagentTool seam, so:

- the #114 consult ladder is **PRESERVED, with its mediation seam moved**. The
  worker still calls ``research_best_practices`` / ``consult_advisor``, which
  lower to :meth:`ToolOutputConsult.consult`. With no ``SubagentTool`` to mediate,
  the worker-leaf consult propagates all the way up to a top-level
  :class:`RunResultConsult`, and the HOST run loop mediates it — routing by
  ``kind`` to a helper harness with a per-kind budget + overflow policy
  (``research`` → web_search, budget 5, SoftFail; ``advice`` → cloud advisor,
  budget 3, EscalateToHuman). Identical #114 semantics, host-owned budgets.
- ``load_skill`` is **dropped** — there is no worker-side per-node seam;
- the ``audit`` skill is **kept**, but now rides the single GLOBAL
  :class:`SkillInjectingContextManager` (the harness's ``context_manager``),
  seeded ALWAYS-ACTIVE at startup. The audit procedure reaches the model
  structurally every turn, compaction-proof, with no ``load_skill`` round-trip.

The tree is DATA: we read the canonical fixture
``fixtures/strategy/cordyceps_tree.json`` and deserialize it — so this example
proves the canonical fixture deserializes and runs.

Run it::

    ollama serve &
    ollama pull gemma4:e4b
    export SPORE_WEB_SEARCH_ENDPOINT=http://localhost:8888/search?format=json
    uv run main.py        # press enter to accept the default audit prompt
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from pydantic import TypeAdapter

from spore_core import (
    Agent,
    AgentId,
    BudgetLimits,
    ConsultHandlerEntry,
    ConsultOverflowPolicyEscalateToHuman,
    ConsultOverflowPolicySoftFail,
    ConsultRequest,
    ConsultResponseAnswer,
    ConsultResponseBudgetExhausted,
    EmptyToolRegistry,
    EscalationActionContinueWithBudget,
    EscalationActionFail,
    EscalationActionSkip,
    EscalationModeSurfaceToHuman,
    EvaluatorResponseVerifier,
    ExecutionRegistry,
    Harness,
    HarnessBuilder,
    HarnessRunOptions,
    HumanRequestBudgetExhausted,
    HumanResponseEscalate,
    HumanResponseHalt,
    InMemoryStorageProvider,
    LoopStrategy,
    ModelAgent,
    NullCacheProvider,
    OllamaModelInterface,
    PausedState,
    ReactConfig,
    RunResultConsult,
    RunResultFailure,
    RunResultSuccess,
    RunResultWaitingForHuman,
    SessionId,
    StandardContextManager,
    StorageProvider,
    Task,
    Verifier,
    WorkspaceConfig,
    WorkspaceScopedSandbox,
    new_session_id,
)
from spore_core.compaction_adapter import into_harness_adapter
from spore_core.harness import ContextManager as HarnessContextManager
from spore_core.harness import loop_strategy_max_steps
from spore_tools import StandardTool, StandardTools, WebSearchTool
from spore_tools.tools.web import SearchMethod, WebSearchConfig

from skills import ACTIVE_SKILLS_KEY, SkillCatalog, SkillInjectingContextManager
from tools import (
    KIND_ADVICE,
    KIND_RESEARCH,
    consult_advisor_tool,
    research_best_practices_tool,
    send_user_message_tool,
)

#: The canonical composed-strategy fixture, read so the example proves the
#: ground-truth tree deserializes (and runs) verbatim — never hand-built.
_TREE_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "strategy" / "cordyceps_tree.json"

#: The verifier registry key the ``SelfVerifying`` node's ``evaluator`` resolves
#: to.
EXEC_EVALUATOR_KEY = "exec-evaluator"

_LOOP_STRATEGY_ADAPTER: TypeAdapter[LoopStrategy] = TypeAdapter(LoopStrategy)

#: The pre-filled audit prompt (press enter to accept).
DEFAULT_AUDIT_PROMPT = (
    "Audit this repository for Rust defects. Discover the crates and their modules, audit each "
    "module for real, actionable defects, and write a markdown report of the most important "
    "findings to `workspace/findings.md`."
)

EXEC_SYSTEM_PROMPT = (
    "You are a cordyceps execution machine. Your strategy is composed declaratively: a Ralph "
    "continuation wrapper drives a PlanExecute, whose plan phase explores the repo and builds a "
    "blocker-aware task graph via `task_list`, and whose execute phase walks that graph as a "
    "ready-set — auditing one module per ready task, each result self-verified by a read-only "
    "evaluator (Default-FAIL: only an explicit PASS clears a task).\n\n"
    "Before each step, call `send_user_message` with one short sentence telling the watching "
    "human what you are about to do and why.\n\n"
    "You are already scoped to the repository root (READ-ONLY). Use `.` for the root and paths "
    "relative to it (e.g. `rust/crates`); never prefix a path with the repository's own folder "
    "name. The audit is read-only — you have no write tool; never attempt to modify source "
    "files.\n\n"
    "Follow the ACTIVE `audit` skill's procedure and output schema exactly: grep first, read "
    "narrow, and return findings as a JSON array of {file, line, severity, description}.\n\n"
    "PLAN phase: explore the repo with `list_dir`/`grep`, then build a blocker-aware task graph "
    "with `task_list` (one task per module; add dependencies where one audit should wait on "
    "another). RALPH wrapper: resume from durable `task_list` progress after each context-window "
    "reset and keep going until every task is done."
)

RESEARCH_PROMPT = (
    "You are a research worker. A peer agent needs factual, current information on a Rust "
    "best-practice or idiom question. Use the web_search tool to gather it, then return a "
    "concise, cited answer as plain text. Act using web_search — do not answer from memory alone."
)

ADVISOR_PROMPT = (
    "You are a senior Rust advisor. A worker is stuck on whether a finding is a real defect, or "
    "on how to rank its severity. Use `read_file` and `grep` to examine the specific code in "
    "question. Then make a decisive call: is this a real defect, what is the severity "
    "(low / medium / high / critical), and why. Be decisive. State your verdict in one sentence, "
    "your reasoning in two. Do not hedge."
)


def _plan_schema() -> dict[str, object]:
    """``plan-schema`` — the task-graph contract the plan phase's ReAct emits."""
    return {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "Ordered task-graph entries; each names a module to audit.",
                "items": {
                    "type": "object",
                    "properties": {
                        "module": {"type": "string", "description": "Module path to audit."},
                        "blockers": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "1-based ids of tasks this one waits on.",
                        },
                    },
                    "required": ["module"],
                },
            },
            "rationale": {"type": "string"},
        },
        "required": ["tasks"],
    }


def _worker_schema() -> dict[str, object]:
    """``worker-schema`` — the per-module finding contract the worker ReAct
    emits."""
    return {
        "type": "array",
        "description": "Findings for ONE module.",
        "items": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path relative to the repo root."},
                "line": {"type": "integer", "description": "1-based line of the defect."},
                "severity": {"enum": ["low", "medium", "high", "critical"]},
                "description": {"type": "string", "description": "Concrete, actionable defect."},
            },
            "required": ["file", "line", "severity", "description"],
        },
    }


def _plan_tools() -> list[StandardTool]:
    """The ``plan-tools`` catalogue: explore + author the task graph
    (read-only)."""
    return [StandardTools.list_dir(), StandardTools.grep(), StandardTools.task_list()]


def _exec_tools() -> list[StandardTool]:
    """The ``exec-tools`` catalogue: read-only audit + the #114 consult ladder +
    human observability. The two consult tools lower to
    :class:`ToolOutputConsult`, which the host run loop mediates (the seam moved
    off ``SubagentTool``)."""
    return [
        StandardTools.read_file(),
        StandardTools.grep(),
        research_best_practices_tool(),
        consult_advisor_tool(),
        send_user_message_tool("🤖"),
    ]


def _build_web_search(endpoint: str) -> StandardTool:
    """Build the SearXNG-backed ``web_search`` catalogue tool (identical to
    06/11)."""
    config = WebSearchConfig(
        endpoint=endpoint,
        method=SearchMethod.GET,
        query_param="q",
        auth_headers=[],
        body_auth_params=[],
    )
    return StandardTool(WebSearchTool.with_config(config), WebSearchTool.schema())


def _build_research_harness(model_id: str, base_url: str, endpoint: str) -> Harness:
    """Build the research handler harness (web_search only) — the
    ``kind="research"`` consult handler. Run host-side on a
    :class:`ConsultRequest`."""
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    return (
        HarnessBuilder.conversational(model)
        .tool(_build_web_search(endpoint))
        .system_prompt(RESEARCH_PROMPT)
        .build()
    )


def _build_advisor_harness(
    model_id: str, base_url: str, repo_sandbox: WorkspaceScopedSandbox
) -> Harness:
    """Build the advisor handler harness (cloud model, read_file + grep) — the
    ``kind="advice"`` consult handler. Rides the same Ollama endpoint via
    ``with_base_url``; only the model id differs (heterogeneous models)."""
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    return (
        HarnessBuilder.conversational(model)
        .sandbox(repo_sandbox)
        .tool(StandardTools.read_file())
        .tool(StandardTools.grep())
        .system_prompt(ADVISOR_PROMPT)
        .build()
    )


def _build_consult_handlers(
    research: Harness, advisor: Harness
) -> dict[str, ConsultHandlerEntry]:
    """Build the HOST-owned ``kind → {handler, budget, overflow}`` map (#114). The
    composed tree has no ``SubagentTool``, so the host run loop holds these
    entries and mediates each :class:`RunResultConsult` against them — the
    per-kind budget lives for the whole run (see ``_mediate_consult``)."""
    return {
        KIND_RESEARCH: ConsultHandlerEntry(
            handler=research, budget=5, overflow=ConsultOverflowPolicySoftFail()
        ),
        KIND_ADVICE: ConsultHandlerEntry(
            handler=advisor, budget=3, overflow=ConsultOverflowPolicyEscalateToHuman()
        ),
    }


def _model_agent(idv: str, model_id: str, base_url: str) -> Agent:
    """Build a model agent over the local Ollama model."""
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    return ModelAgent(AgentId(idv), model)


def exec_evaluator() -> Verifier:
    """The Default-FAIL self-verification evaluator registered under
    ``exec-evaluator``. A single read-only turn (``max_iterations = 1``); the
    neither-pattern ⇒ Failed contract is built into
    :class:`EvaluatorResponseVerifier`."""
    return EvaluatorResponseVerifier(r"(?i)\bPASS\b", r"(?i)\bFAIL\b", 1)


def build_registry(model_id: str, base_url: str) -> ExecutionRegistry:
    """Assemble the :class:`ExecutionRegistry` the cordyceps tree's handles
    resolve against: agents ``planner``/``executor``/``ralph-agent``, toolsets
    ``plan-tools``/``exec-tools``, schemas ``plan-schema``/``worker-schema``, and
    the ``exec-evaluator`` verifier. The handle STRINGS are ground truth from the
    fixture; this is the host-side wiring of those strings to collaborators."""
    return (
        ExecutionRegistry.builder()
        .agent("planner", _model_agent("planner", model_id, base_url))
        .agent("executor", _model_agent("executor", model_id, base_url))
        .agent("ralph-agent", _model_agent("ralph-agent", model_id, base_url))
        # The toolset HANDLES must resolve for `validate()`. Per-node toolset
        # scoping is now RESOLVED (Issue 2): each leaf dispatches its own toolset's
        # tools, wired per-toolset on the HarnessBuilder via `.toolset_tools(...)`
        # (see `main`). These registry slots are validation-only presence entries —
        # never dispatched — so a no-op `EmptyToolRegistry` is sufficient; the real
        # dispatchable catalogues live in `HarnessConfig.toolset_catalogues`.
        .toolset("plan-tools", EmptyToolRegistry())
        .toolset("exec-tools", EmptyToolRegistry())
        .schema("plan-schema", _plan_schema())
        .schema("worker-schema", _worker_schema())
        .verifier(EXEC_EVALUATOR_KEY, exec_evaluator())
        .build()
    )


async def build_global_context_manager(
    model_id: str,
    base_url: str,
    storage: StorageProvider,
    repo_root: Path,
    bundled_audit: str,
    session: SessionId,
) -> HarnessContextManager:
    """Build the GLOBAL skill-injecting context manager with the ``audit`` skill
    seeded ALWAYS-ACTIVE for ``session``. Wraps the standard compaction
    adapter."""
    catalog = await SkillCatalog.bootstrap(repo_root, bundled_audit)
    # Seed `audit` always-active: the global context_manager injects its body
    # structurally every turn (no `load_skill` round-trip in the composed tree).
    await storage.run().put(session, ACTIVE_SKILLS_KEY, ["audit"])

    inner_model = OllamaModelInterface.with_base_url(model_id, base_url)
    inner = into_harness_adapter(StandardContextManager(inner_model, NullCacheProvider()))
    return SkillInjectingContextManager(inner, storage.run(), catalog.manifest())


def build_task(prompt: str, session: SessionId) -> Task:
    """Build the cordyceps :class:`Task`: the composed tree deserialized from the
    fixture, under a generous global backstop so the per-node ``PerLoop{12}``
    worker bound fires first."""
    tree = _LOOP_STRATEGY_ADAPTER.validate_json(_TREE_PATH.read_text(encoding="utf-8"))
    return Task.new(prompt, session, tree, budget=BudgetLimits(max_turns=64))


def _load_tree() -> LoopStrategy:
    return _LOOP_STRATEGY_ADAPTER.validate_json(_TREE_PATH.read_text(encoding="utf-8"))


def _truncate(text: str, limit: int = 280) -> str:
    """Keep boundary lines readable — findings and reports can be long."""
    flat = text.replace("\n", " ")
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


def _prompt_line(prompt: str) -> str:
    """Print a prompt and read one line from stdin (trailing newline stripped)."""
    try:
        return input(prompt)
    except EOFError:
        return ""


def _read_audit_prompt() -> str | None:
    """Read one audit prompt from the REPL. The prompt string to run (empty line
    ⇒ the default verbatim); ``None`` on EOF (Ctrl-D), which quits the REPL."""
    print("Default audit prompt (press enter to accept, or type your own; Ctrl-D to quit):")
    print(f"  {DEFAULT_AUDIT_PROMPT}")
    try:
        line = input("audit> ")
    except EOFError:
        return None
    return DEFAULT_AUDIT_PROMPT if not line.strip() else line


def _render_consult_instruction(request: ConsultRequest) -> str:
    """Render a :class:`ConsultRequest` to the handler's instruction text
    (#114)."""
    return (
        f"A worker agent is requesting help (kind: {request.kind}).\n\n"
        f"Situation: {request.situation}\n\n"
        f"Attempts so far: {request.attempts}\n\n"
        f"Question: {request.question}"
    )


async def _mediate_consult(
    harness: Harness,
    handlers: dict[str, ConsultHandlerEntry],
    counts: dict[str, int],
    request: ConsultRequest,
    state: PausedState,
) -> object:
    """Mediate one worker leaf consult HOST-SIDE (#114, seam relocated off
    ``SubagentTool``). Routes by ``kind``, enforces the per-kind budget held in
    ``counts`` for the whole run, runs the handler harness as a direct child, and
    resumes the paused composed tree with the answer — or applies the overflow
    policy (``SoftFail`` resumes with ``BudgetExhausted``; ``EscalateToHuman``
    surfaces the advisor ladder to the operator). Identical to the old
    ``SubagentTool`` mediation, only the owner moved."""
    entry = handlers.get(request.kind)
    if entry is None:
        # No handler for this kind ⇒ resume the worker without help (loud, not a
        # silent stall). Matches the SubagentTool R6 graceful degradation.
        print(f"\n(no consult handler for kind {request.kind!r}; worker proceeds)", file=sys.stderr)
        return await harness.resume_consult(
            state,
            ConsultResponseBudgetExhausted(
                message=(
                    f"no consult handler for kind {request.kind!r}; "
                    "proceed without further help"
                ),
            ),
        )

    # Per-kind budget: `used` is how many consults of this kind were already
    # mediated this run. The handler runs while `used < budget`; the (budget+1)th
    # consult overflows.
    used = counts.get(request.kind, 0)
    if used >= entry.budget:
        if isinstance(entry.overflow, ConsultOverflowPolicySoftFail):
            print(
                f"\n(consult budget for {request.kind!r} exhausted — "
                "worker finishes with what it has)"
            )
            return await harness.resume_consult(
                state,
                ConsultResponseBudgetExhausted(
                    message=(
                        f"consult budget for kind {request.kind!r} exhausted; "
                        "proceed without further help"
                    ),
                ),
            )
        return await _handle_consult_overflow(harness, entry, request, state)

    # Run the handler harness as a direct child (depth-1, never under the worker)
    # on the consult rendered to text, then resume with its answer.
    counts[request.kind] = used + 1
    print(f"\n┌─ consult ({request.kind}) → {used + 1} of {entry.budget} budget")
    instruction = _render_consult_instruction(request)
    task = Task.new(instruction, new_session_id(), ReactConfig.per_loop(16))
    result = await entry.handler.run(HarnessRunOptions(task))
    if isinstance(result, RunResultSuccess):
        answer = result.output
    else:
        # A handler that does not cleanly complete must not stall the worker —
        # feed its failure text back as the answer so the worker can adapt.
        answer = f"consult handler did not complete cleanly: {result!r}"
    print(f"└─ consult answer: {_truncate(answer, 200)}")
    return await harness.resume_consult(state, ConsultResponseAnswer(text=answer))


async def _handle_consult_overflow(
    harness: Harness,
    entry: ConsultHandlerEntry,
    request: ConsultRequest,
    state: PausedState,
) -> object:
    """The ``advice`` consult overflowed its budget under ``EscalateToHuman``:
    present the #114 three-choice ladder to the operator and resume the worker
    with the decision. Preserves the original ladder semantics host-side."""
    print("\n╔═ HUMAN ESCALATION (advisor budget exhausted) ═")
    print(f"║ situation: {_truncate(request.situation, 200)}")
    print(f"║ question : {_truncate(request.question, 200)}")
    print("║ [1] run the advisor once more (host-side)")
    print("║ [2] abort this consult — worker proceeds without help")
    print("║ [3] type a free-form answer yourself")
    print("╚═════════════════════════════════════════════════")

    choice = _prompt_line("> ").strip()
    if choice == "2":
        return await harness.resume_consult(
            state,
            ConsultResponseBudgetExhausted(
                message="advisor budget exhausted; proceed without further help",
            ),
        )
    if choice == "3":
        text = _prompt_line("answer> ")
        return await harness.resume_consult(state, ConsultResponseAnswer(text=text))

    # Default ([1] or empty): run the advisor handler once more host-side and
    # inject its answer — a bounded escape hatch past the per-kind budget.
    print("(running advisor for one more turn…)")
    task = Task.new(_render_consult_instruction(request), new_session_id(), ReactConfig.per_loop(16))
    result = await entry.handler.run(HarnessRunOptions(task))
    answer = result.output if isinstance(result, RunResultSuccess) else (
        f"advisor did not complete cleanly: {result!r}"
    )
    print(f"advisor: {_truncate(answer, 300)}")
    return await harness.resume_consult(state, ConsultResponseAnswer(text=answer))


def _describe_action(action: object) -> str:
    if isinstance(action, EscalationActionContinueWithBudget):
        return f"continue with +{action.steps} steps"
    if isinstance(action, EscalationActionSkip):
        return "skip this task"
    if isinstance(action, EscalationActionFail):
        return "fail this node"
    return repr(action)


async def _handle_human_escalation(
    harness: Harness, state: PausedState, request: object
) -> object:
    """Present a ``BudgetExhausted`` pause and resume with the operator's choice.
    The composed tree surfaces a runaway node here under ``SurfaceToHuman``; we
    offer its ``available_actions`` and resume by re-resolving handles (no
    reconfiguration)."""
    if not isinstance(request, HumanRequestBudgetExhausted):
        # The composed tree only escalates via BudgetExhausted; anything else is
        # unexpected — halt cleanly.
        print(f"\nunexpected human request: {request!r}", file=sys.stderr)
        return await harness.resume(state, HumanResponseHalt())

    phase, actions = request.phase, request.available_actions
    print(f"\n╔═ BUDGET ESCALATION ({phase}) ═══════════════════")
    for i, a in enumerate(actions):
        print(f"║ [{i + 1}] {_describe_action(a)}")
    print("╚═════════════════════════════════════════════════")

    choice = _prompt_line("> ").strip()
    try:
        idx = int(choice) - 1
    except ValueError:
        idx = 0
    if 0 <= idx < len(actions):
        action = actions[idx]
    else:
        # Default to a small budget bump so an empty line keeps the run going.
        action = EscalationActionContinueWithBudget(steps=12)

    print(f"(resuming with {_describe_action(action)})")
    return await harness.resume(state, HumanResponseEscalate(action=action))


async def main() -> int:
    parser = argparse.ArgumentParser(description="spore-core cordyceps composed-strategy auditor")
    parser.add_argument("--model")
    parser.add_argument("--advisor-model")
    parser.add_argument("--search-url")
    args = parser.parse_args()

    model_id = args.model or os.environ.get("SPORE_OLLAMA_MODEL") or "gemma4:e4b"
    # The #114 advisor consult handler runs a heterogeneous (cloud) model.
    advisor_model_id = (
        args.advisor_model or os.environ.get("SPORE_ADVISOR_MODEL") or "minimax-m3:cloud"
    )
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)

    # The research consult handler needs a SearXNG JSON endpoint — fail fast (like
    # examples 06/11) so a missing backend is a startup error, not a mid-run
    # surprise.
    endpoint = (args.search_url or os.environ.get("SPORE_WEB_SEARCH_ENDPOINT") or "").strip()
    if not endpoint:
        print(
            "SPORE_WEB_SEARCH_ENDPOINT is not set.\n"
            "Set it to a SearXNG JSON endpoint, e.g. "
            "http://localhost:8888/search?format=json. See .env.example and the README.",
            file=sys.stderr,
        )
        return 2

    repo_root = Path.cwd().resolve(strict=True)
    workspace_root = (Path(__file__).parent / "workspace").resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    bundled_audit = (Path(__file__).parent / "skills" / "audit" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    # AC5: the fully-bounded tree's worst-case per-window turn count is computable
    # BEFORE the run. Ralph[PlanExecute[ReAct{4}, SelfVerifying[ReAct{12}]]]
    # = 4 + (12 + 1) = 17. An `Unlimited` anywhere would collapse this to None.
    tree_preview = _load_tree()
    print(f"model        : {model_id}")
    print(f"advisor model: {advisor_model_id}")
    print(f"search       : {endpoint}")
    print(f"repo root    : {repo_root}")
    print("strategy     : Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]] (from fixture)")
    print(
        f"max_steps    : {loop_strategy_max_steps(tree_preview)}  "
        "(per-window worst case; Unlimited anywhere ⇒ None)"
    )
    print(
        "consults     : research(web_search, budget 5, soft-fail), "
        "advice(advisor, budget 3, escalate)"
    )
    print()

    while True:
        prompt = _read_audit_prompt()
        if prompt is None:
            break

        session = new_session_id()
        storage = StorageProvider.single(InMemoryStorageProvider())

        # Read-only repo sandbox: the audit never writes source files.
        sandbox = WorkspaceScopedSandbox(WorkspaceConfig(root=repo_root, read_only=True))
        # The advisor handler gets its own read-only view of the repo (read_file +
        # grep) so it can inspect the code the worker is asking about.
        advisor_sandbox = WorkspaceScopedSandbox(WorkspaceConfig(root=repo_root, read_only=True))

        # The HOST-owned consult ladder (#114). The seam moved off `SubagentTool`:
        # the host loop holds these handlers + per-kind budgets for the whole run.
        consult_handlers = _build_consult_handlers(
            _build_research_harness(model_id, base_url, endpoint),
            _build_advisor_harness(advisor_model_id, base_url, advisor_sandbox),
        )

        registry = build_registry(model_id, base_url)
        context_manager = await build_global_context_manager(
            model_id, base_url, storage, repo_root, bundled_audit, session
        )

        # The harness's own model drives the Ralph wrapper; the per-node agents
        # come from the registry. Compaction/summarization uses this model too.
        model = OllamaModelInterface.with_base_url(model_id, base_url)
        # Issue 2 (per-node toolset scoping): each leaf dispatches ONLY its own
        # toolset. The real tools are wired per-toolset — `plan-tools` for the
        # planner node, `exec-tools` for the worker node — so the planner cannot
        # reach an exec-only tool (`read_file`) and the worker cannot reach a
        # plan-only tool (`task_list`/`list_dir`). The leaf's `toolset` handle on
        # the serialized tree drives the lookup.
        harness = (
            HarnessBuilder.conversational(model)
            .sandbox(sandbox)
            .storage(storage)
            .registry(registry)
            .escalation_mode(EscalationModeSurfaceToHuman())
            .system_prompt(EXEC_SYSTEM_PROMPT)
            .context_manager(context_manager)
            .toolset_tools("plan-tools", _plan_tools())
            .toolset_tools("exec-tools", _exec_tools())
            .build()
        )

        task = build_task(prompt, session)
        # Per-RUN consult counts (host-owned, #114): how many consults of each
        # `kind` have already been mediated. Persists across every pause/resume of
        # THIS audit so the per-kind budget bounds the whole run, not one turn.
        consult_counts: dict[str, int] = {}
        try:
            result: object = await harness.run(HarnessRunOptions(task))
        except OSError as e:
            print(
                f"\ncould not reach the model — is `ollama serve` running? ({e})", file=sys.stderr
            )
            return 1

        while True:
            if isinstance(result, RunResultSuccess):
                print(f"\ndone ({result.turns} turn(s)): {_truncate(result.output, 400)}")
                break
            if isinstance(result, RunResultFailure):
                print(
                    f"\nfailed after {result.turns} turn(s): {result.reason!r}", file=sys.stderr
                )
                break
            if isinstance(result, RunResultConsult):
                # A worker leaf consult propagated up through the composed tree
                # (no SubagentTool to absorb it). The host mediates.
                result = await _mediate_consult(
                    harness, consult_handlers, consult_counts, result.request, result.state
                )
                continue
            if isinstance(result, RunResultWaitingForHuman):
                result = await _handle_human_escalation(harness, result.state, result.request)
                continue
            print(f"\nrun ended unexpectedly: {result!r}", file=sys.stderr)
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
