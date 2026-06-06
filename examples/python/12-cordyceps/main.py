"""spore-core example 12 — cordyceps: a fully autonomous task-completion agent.

The thesis: you give it a task; it does not stop until the job is done — and when
a worker gets stuck or uncertain, it asks for help (a sibling helper, then a
human) rather than giving up.

This example composes everything the suite has built — subagents-as-tools (11),
custom sandboxed tools (05), ``web_search`` (06), ``memory`` (07), ``task_list``
— plus two new capabilities:

- **Architect-side skill loading** (see :mod:`skills`): a ``load_skill`` tool
  activates the bundled ``audit`` skill at runtime via a ``GuideRegistry``, and a
  custom context manager re-injects the skill body every turn (compaction-proof).
  This is the pattern issue #115 will absorb into the harness; the live loop's
  structural skill-injection path is not wired yet (see the README and #115).
- **A generalized consult / escalation ladder** (issue #114): the analysis
  worker escalates mid-loop to a research helper (``kind=research``, budget 5,
  soft-fail) and then to a cloud-model advisor (``kind=advice``, budget 3,
  escalate-to-human), resuming each time without ending its run.

Topology (depth-1)::

    orchestrator (ReAct, gemma4:e4b)
      tools: list_dir, grep, task_list, memory, write_file, bash_command,
             analysis_worker (SubagentTool, with consult handlers)
      ├── analysis_worker (Isolated) — audits ONE module
      │     tools: read_file, grep, research_best_practices, consult_advisor,
      │            load_skill
      ├── research_worker (Isolated) — web_search   [consult handler: research]
      └── advisor         (Isolated, cloud model)   [consult handler: advice]

The orchestrator enumerates crates → modules, adds one ``task_list`` task per
module, dispatches the analysis worker per task, accumulates findings in
``memory``, finalizes the top 5, writes ``workspace/findings.md``, and runs the
y/N issue-filing flow. The audit is READ-ONLY; the only writes are
``workspace/findings.md`` and (approved) GitHub issues.

Run it::

    ollama serve &
    ollama pull gemma4:e4b
    export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
    uv run main.py
    # press enter at the prompt to accept the default audit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from spore_core import (
    BudgetLimits,
    ConsultHandlerEntry,
    ConsultOverflowPolicyEscalateToHuman,
    ConsultOverflowPolicySoftFail,
    Harness,
    HarnessBuilder,
    HarnessRunOptions,
    HumanRequestClarification,
    HumanRequestReview,
    HumanResponseAnswer,
    HumanResponseHalt,
    InMemoryStorageProvider,
    LoopStrategyReAct,
    NullCacheProvider,
    OllamaModelInterface,
    RegistryToolSchema,
    RunResultFailure,
    RunResultSuccess,
    RunResultWaitingForHuman,
    StandardContextManager,
    StorageProvider,
    StreamToolCall,
    StreamToolResult,
    StreamTurnStart,
    Task,
    ToolAnnotations,
    WorkspaceConfig,
    WorkspaceScopedSandbox,
    new_session_id,
)
from spore_core.compaction_adapter import into_harness_adapter
from spore_tools import StandardTool, StandardTools, WebSearchTool
from spore_tools.tools.subagent import ContextSharingIsolated, SubagentTool
from spore_tools.tools.web import SearchMethod, WebSearchConfig

from skills import SkillCatalog, SkillInjectingContextManager
from tools import (
    KIND_ADVICE,
    KIND_RESEARCH,
    consult_advisor_tool,
    load_skill_tool,
    research_best_practices_tool,
)

# The pre-filled audit prompt the user presses enter to accept (from the issue).
# An empty line at the REPL ⇒ this verbatim.
DEFAULT_AUDIT_PROMPT = (
    "Audit the current repo for the rust language. Work sequentially by identifying each crate, "
    "and each module in the crate, and adding a task to the tasklist for a subagent to do the "
    "deep dive audit on."
)

# Per-worker wall-clock cap (seconds). A worker can burn many internal ReAct
# turns (and mediated consults); this bounds how long the orchestrator waits on
# any single delegation.
WORKER_TIMEOUT_SECONDS = 300.0

# The tool name that maps to the analysis-worker subagent (vs. the standard
# tools). Used to decide which stream events get the boxed boundary banner.
ANALYSIS_WORKER = "analysis_worker"

ORCHESTRATOR_PROMPT = (
    "You are the cordyceps orchestrator: an autonomous Rust-repo auditor. You do not stop until "
    "the audit is complete. Work sequentially: (1) use `list_dir` to enumerate the crates under "
    "`rust/crates/`, then the modules (`src/*.rs`) in each crate; (2) for each module, add ONE "
    "task to the task list (`task_list`) describing the module to audit; (3) for each task, call "
    "`analysis_worker` with an `instruction` naming the ONE module to deep-dive audit; (4) "
    "accumulate the findings each worker returns into `memory` under a stable key; (5) when every "
    "module is audited, pick the TOP 5 most important findings across all modules and write them "
    "as a markdown report to `workspace/findings.md` using `write_file`. The audit is READ-ONLY — "
    "never modify source files. Delegate the per-module deep dives to `analysis_worker`; do not "
    "audit modules yourself. Finish by writing findings.md."
)

ANALYSIS_WORKER_PROMPT = (
    "You are an analysis worker: you deep-dive audit exactly ONE Rust module for real, actionable "
    'defects. BEFORE auditing, call `load_skill` with `skill_id` = "audit" and follow the '
    "returned procedure and findings schema EXACTLY. Stay inside the one module you were given. "
    "Grep first, read only narrow line ranges, and escalate with `research_best_practices` (idiom "
    "questions) or `consult_advisor` (severity / is-this-real questions) when genuinely unsure. "
    "Your FINAL answer must be a JSON array of {file, line, severity, description} objects — and "
    "nothing else."
)

RESEARCH_PROMPT = (
    "You are a research worker. Use the web_search tool to gather current, factual information on "
    "the Rust best-practice or idiom question you are given. Issue focused queries, read the "
    "results, and return a concise, cited answer as plain text. Act using web_search — do not "
    "answer from memory alone."
)

ADVISOR_PROMPT = (
    "You are a senior Rust advisor. A worker is stuck on whether a finding is a real defect, or on "
    "how to rank its severity. Use `read_file` and `grep` to investigate the specific code in "
    "question, then give a crisp, decisive recommendation: is it a real defect, what severity "
    "(low/medium/high/critical), and why. Be concrete and brief."
)


def _instruction_schema() -> dict[str, object]:
    """The single-parameter input schema every subagent tool advertises."""
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


def _build_inner_context_manager(model_id: str, base_url: str):  # type: ignore[no-untyped-def]
    """Build the inner standard compaction adapter (the same one
    :meth:`HarnessBuilder.conversational` installs), so the skill-injecting
    context manager can wrap it and delegate every non-``assemble`` method to
    it."""
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    return into_harness_adapter(StandardContextManager(model, NullCacheProvider()))


def _build_research_harness(model_id: str, base_url: str, endpoint: str) -> Harness:
    """Build the research worker harness (web_search only). Used as the
    ``kind=research`` consult handler."""
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
    """Build the advisor harness (cloud model, read_file + grep). Used as the
    ``kind=advice`` consult handler. Rides the same Ollama endpoint via
    ``with_base_url``; only the model id differs."""
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    return (
        HarnessBuilder.conversational(model)
        .sandbox(repo_sandbox)
        .tool(StandardTools.read_file())
        .tool(StandardTools.grep())
        .system_prompt(ADVISOR_PROMPT)
        .build()
    )


def _build_analysis_harness(
    model_id: str,
    base_url: str,
    repo_sandbox: WorkspaceScopedSandbox,
    storage: StorageProvider,
    catalog: SkillCatalog,
) -> Harness:
    """Build the analysis worker harness: read_file, grep, the two consult tools,
    and load_skill. Isolated; audits ONE module.

    The worker shares the SAME storage as the orchestrator so ``load_skill``'s
    active-skill write and the context manager's read rendezvous within the run.
    Subagents run Isolated SESSIONS, but the run_store is keyed by the worker's
    own session id, so each worker activates ``audit`` for itself."""
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    inner_cm = _build_inner_context_manager(model_id, base_url)
    skill_cm = SkillInjectingContextManager(inner_cm, storage.run(), catalog.manifest())
    return (
        HarnessBuilder.conversational(model)
        .sandbox(repo_sandbox)
        .storage(storage)
        .context_manager(skill_cm)
        .tool(StandardTools.read_file())
        .tool(StandardTools.grep())
        .tool(research_best_practices_tool())
        .tool(consult_advisor_tool())
        .tool(load_skill_tool(catalog.registry()))
        .system_prompt(ANALYSIS_WORKER_PROMPT)
        .build()
    )


def _build_analysis_tool(
    analysis: Harness, consult_handlers: dict[str, ConsultHandlerEntry]
) -> StandardTool:
    """Wrap the analysis worker as a ``SubagentTool`` with the consult handlers
    installed. The two handlers mediate by ``kind`` (research → research_worker,
    advice → advisor) with the per-kind budgets + overflow policies from #114."""
    from spore_core import StandardToolRegistry

    empty_child_registry = StandardToolRegistry()
    subagent = SubagentTool.new(
        name=ANALYSIS_WORKER,
        description=(
            "Delegate a deep-dive audit of ONE Rust module: pass an `instruction` naming the "
            "module; it loads the `audit` skill, audits the module (escalating via consults when "
            "stuck), and returns a JSON array of {file, line, severity, description} findings."
        ),
        input_schema=_instruction_schema(),
        timeout_seconds=WORKER_TIMEOUT_SECONDS,
        context_sharing=ContextSharingIsolated(),
        harness=analysis,
        child_registry=empty_child_registry,
    ).with_consult_handlers(consult_handlers)

    schema = RegistryToolSchema(
        name=ANALYSIS_WORKER,
        description="Deep-dive audit one module via a subagent; returns JSON findings.",
        parameters=_instruction_schema(),
        annotations=ToolAnnotations(open_world=True),
    )
    return StandardTool(subagent, schema)


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


def _read_audit_prompt() -> str:
    """Read the audit prompt from the REPL: print the default, accept an empty
    line as the default verbatim."""
    print("Default audit prompt (press enter to accept, or type your own):")
    print(f"  {DEFAULT_AUDIT_PROMPT}")
    line = _prompt_line("audit> ")
    return DEFAULT_AUDIT_PROMPT if not line.strip() else line


def _seed_bundled_audit_skill(repo_root: Path, bundled_audit: str) -> None:
    """Seed ``.spore/skills/audit/SKILL.md`` from the bundled copy if absent, so
    a user can see the filesystem-registry shape (documented in the README)."""
    directory = repo_root / ".spore" / "skills" / "audit"
    file = directory / "SKILL.md"
    if file.exists():
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        file.write_text(bundled_audit, encoding="utf-8")
    except OSError:
        pass


def _make_stream_sink(call_names: dict[str, str]):  # type: ignore[no-untyped-def]
    """Build the orchestrator stream sink: boundary banners for the analysis
    worker, terse lines for the standard tools (mirrors 11-multi-agent)."""

    def on_stream(event: object) -> None:
        if isinstance(event, StreamTurnStart):
            print(f"orchestrator · turn {event.turn}")
        elif isinstance(event, StreamToolCall):
            call_names[event.call_id] = event.name
            if event.name == ANALYSIS_WORKER:
                instruction = event.args.get("instruction", "<no instruction>")
                print(f"┌─ orchestrator → {ANALYSIS_WORKER}")
                print(f"│  received: {_truncate(str(instruction), 200)}")
            else:
                print(f"  orchestrator → {event.name}({_truncate(json.dumps(event.args), 140)})")
        elif isinstance(event, StreamToolResult):
            name = call_names.pop(event.call_id, "<tool>")
            if name == ANALYSIS_WORKER:
                tag = "FAILED" if event.is_error else "findings"
                print(f"└─ {ANALYSIS_WORKER} → orchestrator")
                print(f"   {tag}: {_truncate(event.content, 300)}")
            else:
                tag = "err" if event.is_error else "ok"
                print(f"  {name} → orchestrator [{tag}]: {_truncate(event.content, 140)}")

    return on_stream


async def _handle_human_escalation(
    orchestrator: Harness,
    advisor_handler: Harness,
    state: object,
    request: object,
) -> object:
    """Handle an advice-budget-exhausted human escalation with the three-choice
    ladder. Returns the next ``RunResult`` (the orchestrator resumed).

    IMPORTANT (honest mechanics): the worker's paused consult lives inside the
    orchestrator's ``PausedState.child_state``, and the harness does NOT yet wire
    a child-consult resume through the parent (``_resume_inner`` no-ops the
    ``child_state`` branch — that lands with the #5/#115 follow-up). So every
    choice here resumes the ORCHESTRATOR with the human's decision injected as
    guidance; the specific module's in-flight worker audit is dropped. "+1
    advisor turn" re-runs the advisor handler HOST-SIDE and injects its answer as
    that guidance — the closest we can get to a budget bump without a core
    primitive."""
    if isinstance(request, HumanRequestReview):
        context = request.content
    elif isinstance(request, HumanRequestClarification):
        context = request.question
    else:
        context = "tool approval requested"

    print("\n╔═ HUMAN ESCALATION (advisor budget exhausted) ═")
    print(f"║ {_truncate(context, 400)}")
    print("╚═══════════════════════════════════════════════")
    print("Choose: [1] +1 advisor turn  [2] abort subagent & chat  [3] free-form answer")

    choice = _prompt_line("> ").strip()
    if choice == "1":
        # Re-run the advisor handler once, host-side, on the escalation context;
        # inject its output as the orchestrator's guidance.
        print("(running advisor for one more turn…)")
        task = Task.new(context, new_session_id(), LoopStrategyReAct(max_iterations=16))
        advisor_result = await advisor_handler.run(HarnessRunOptions(task))
        if isinstance(advisor_result, RunResultSuccess):
            advice = advisor_result.output
        else:
            advice = f"advisor did not complete cleanly: {advisor_result!r}"
        print(f"advisor: {_truncate(advice, 300)}")
        return await orchestrator.resume(state, HumanResponseAnswer(text=advice))  # type: ignore[arg-type]
    if choice == "2":
        print("(aborting the stuck subagent; returning to the orchestrator…)")
        return await orchestrator.resume(state, HumanResponseHalt())  # type: ignore[arg-type]

    text = _prompt_line("your answer> ") if choice == "3" else choice
    return await orchestrator.resume(state, HumanResponseAnswer(text=text))  # type: ignore[arg-type]


async def _run_issue_filing_flow(orchestrator: Harness, findings: Path) -> None:
    """After a successful audit, present the top-5 and offer to file them as
    issues. The model drives ``gh issue create`` via ``bash_command`` (no ``gh``
    skill)."""
    report = findings.read_text(encoding="utf-8") if findings.exists() else ""
    print("\n── top findings (workspace/findings.md) ──")
    print(_truncate(report, 1200))
    print("──────────────────────────────────────────")

    answer = _prompt_line("File these as GitHub issues? [y/N] ")
    if answer.strip().lower() != "y":
        print("Not filing. Done.")
        return

    print("(asking the orchestrator to file the top 5 via `gh issue create`…)")
    task = Task.new(
        "Using `bash_command`, file the TOP 5 findings from workspace/findings.md as GitHub "
        "issues via `gh issue create` — one issue per finding, with a clear title and a body "
        "containing the file, line, severity, and description. Run `gh` once per finding. Report "
        "the issue URLs when done.",
        new_session_id(),
        LoopStrategyReAct(max_iterations=24),
    )
    result = await orchestrator.run(HarnessRunOptions(task))
    if isinstance(result, RunResultSuccess):
        print(f"\nfiling done: {_truncate(result.output, 400)}")
    else:
        print(f"\nfiling did not complete cleanly: {result!r}", file=sys.stderr)


async def main() -> int:
    parser = argparse.ArgumentParser(description="spore-core cordyceps autonomous auditor")
    parser.add_argument("--model")
    parser.add_argument("--advisor-model")
    parser.add_argument("--search-url")
    args = parser.parse_args()

    model_id = args.model or os.environ.get("SPORE_OLLAMA_MODEL") or "gemma4:e4b"
    advisor_model_id = (
        args.advisor_model or os.environ.get("SPORE_ADVISOR_MODEL") or "minimax-m3:cloud"
    )
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)

    # Required search backend (research worker) — fail fast like 06/11.
    endpoint = (args.search_url or os.environ.get("SPORE_WEB_SEARCH_ENDPOINT") or "").strip()
    if not endpoint:
        print(
            "SPORE_WEB_SEARCH_ENDPOINT is not set.\n"
            "Set it to a SearXNG JSON endpoint, e.g. "
            "http://localhost:8888/search?format=json. See .env.example and the README.",
            file=sys.stderr,
        )
        return 2

    # Resolve the repo root (cwd) for the read-only audit sandbox, and this
    # example's workspace/ for the report write.
    repo_root = Path.cwd().resolve(strict=True)
    workspace_root = (Path(__file__).parent / "workspace").resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    bundled_audit = (Path(__file__).parent / "skills" / "audit" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    # Seed `.spore/skills/audit/SKILL.md` from the bundled copy if absent.
    _seed_bundled_audit_skill(repo_root, bundled_audit)

    # One in-memory storage provider, shared by the orchestrator and the analysis
    # worker so `load_skill` (worker-side write) and the context manager (read)
    # rendezvous on `run_store["active_skills"]`.
    storage = StorageProvider.single(InMemoryStorageProvider())

    # Scan + register skills (bundled audit + any project/user skills).
    catalog = await SkillCatalog.bootstrap(repo_root, bundled_audit)

    # The orchestrator can read the whole repo + write into its own workspace; a
    # workspace-scoped sandbox rooted at the repo lets read_file/grep/list_dir
    # reach the crates, write_file land findings.md, and bash_command run `gh`.
    # For the audit's read-only guarantee we rely on the orchestrator prompt +
    # skill discipline, not a read-only sandbox, because it must also write
    # findings.md and (optionally) run `gh`.
    orchestrator_sandbox = WorkspaceScopedSandbox(WorkspaceConfig(root=repo_root))
    # Workers and the advisor get a READ-ONLY view: same root, but their tool sets
    # (read_file/grep) never write. A dedicated sandbox keeps them scoped.
    worker_sandbox = WorkspaceScopedSandbox(WorkspaceConfig(root=repo_root))

    # ---- Build the consult handlers (research + advice) ---------------------
    research_handler = _build_research_harness(model_id, base_url, endpoint)
    advisor_handler = _build_advisor_harness(advisor_model_id, base_url, worker_sandbox)
    consult_handlers: dict[str, ConsultHandlerEntry] = {
        KIND_RESEARCH: ConsultHandlerEntry(
            handler=research_handler, budget=5, overflow=ConsultOverflowPolicySoftFail()
        ),
        KIND_ADVICE: ConsultHandlerEntry(
            handler=advisor_handler, budget=3, overflow=ConsultOverflowPolicyEscalateToHuman()
        ),
    }

    # ---- Build the analysis worker + wrap it (with consult handlers) --------
    analysis = _build_analysis_harness(model_id, base_url, worker_sandbox, storage, catalog)
    analysis_tool = _build_analysis_tool(analysis, consult_handlers)

    # ---- Build the orchestrator --------------------------------------------
    orchestrator_model = OllamaModelInterface.with_base_url(model_id, base_url)
    orchestrator = (
        HarnessBuilder.conversational(orchestrator_model)
        .sandbox(orchestrator_sandbox)
        .storage(storage)
        .tool(StandardTools.list_dir())
        .tool(StandardTools.grep())
        .tool(StandardTools.task_list())
        .tool(StandardTools.memory())
        .tool(StandardTools.write_file())
        .tool(StandardTools.bash_command())
        .tool(analysis_tool)
        .system_prompt(ORCHESTRATOR_PROMPT)
        .build()
    )

    skill_names = ", ".join(e.name for e in catalog.manifest())
    print(f"model        : {model_id}")
    print(f"advisor model: {advisor_model_id}")
    print(f"endpoint     : {endpoint}")
    print(f"repo root    : {repo_root}")
    print(f"workspace    : {workspace_root}")
    print(f"skills       : {skill_names}")
    print("strategy     : orchestrator=ReAct, workers=ReAct (isolated)\n")

    # ---- REPL: pre-filled prompt, enter accepts the default -----------------
    prompt = _read_audit_prompt()

    call_names: dict[str, str] = {}
    task = Task.new(
        prompt,
        new_session_id(),
        LoopStrategyReAct(max_iterations=64),
        budget=BudgetLimits(max_turns=64),
    )
    options = HarnessRunOptions(task, on_stream=_make_stream_sink(call_names))

    # ---- Drive the orchestrator, handling the human-escalation ladder -------
    try:
        result = await orchestrator.run(options)
    except OSError as e:
        print(f"\ncould not reach the model — is `ollama serve` running? ({e})", file=sys.stderr)
        return 1

    while True:
        if isinstance(result, RunResultSuccess):
            print(f"\norchestrator done ({result.turns} turn(s)): {_truncate(result.output, 400)}")
            findings = workspace_root / "findings.md"
            if findings.exists():
                print(f"\nfindings.md written: {findings}")
                await _run_issue_filing_flow(orchestrator, findings)
            else:
                print(
                    "\nwarning: orchestrator finished but workspace/findings.md was not written.",
                    file=sys.stderr,
                )
            return 0
        if isinstance(result, RunResultFailure):
            print(
                f"\norchestrator failed after {result.turns} turn(s): {result.reason!r}",
                file=sys.stderr,
            )
            return 1
        if isinstance(result, RunResultWaitingForHuman):
            # The advice consult budget (3) was exhausted under EscalateToHuman:
            # the analysis_worker SubagentTool converted the over-budget consult
            # into a human pause, which bubbled up here.
            result = await _handle_human_escalation(
                orchestrator, advisor_handler, result.state, result.request
            )
            continue
        print(f"\nrun ended unexpectedly: {result!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
