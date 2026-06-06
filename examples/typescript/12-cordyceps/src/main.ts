/**
 * spore-core example 12 — **cordyceps**: a fully autonomous task-completion
 * agent (the capstone).
 *
 * **The thesis: you give it a task; it does not stop until the job is done — and
 * when a worker gets stuck or uncertain, it asks for help (a sibling helper,
 * then a human) rather than giving up.**
 *
 * This example composes everything the suite has built — subagents-as-tools
 * (11), custom sandboxed tools (05), `web_search` (06), `memory` (07),
 * `task_list` — plus two new capabilities:
 *
 * - **Architect-side skill loading** (see `skills.ts`): a `load_skill` tool
 *   activates the bundled `audit` skill at runtime via a `GuideRegistry`, and a
 *   custom context manager re-injects the skill body every turn
 *   (compaction-proof). This is the pattern issue #115 will absorb into the
 *   harness; the live loop's structural skill-injection path is not wired yet
 *   (see the README and #115).
 * - **A generalized consult / escalation ladder** (issue #114): the analysis
 *   worker escalates mid-loop to a research helper (`kind=research`, budget 5,
 *   soft-fail) and then to a cloud-model advisor (`kind=advice`, budget 3,
 *   escalate-to-human), resuming each time without ending its run.
 *
 * ## Topology (depth-1)
 *
 * ```text
 * orchestrator (ReAct, gemma4:e4b)
 *   tools: list_dir, grep, task_list, memory, write_file, bash_command,
 *          analysis_worker (SubagentTool, with consult handlers)
 *   ├── analysis_worker (Isolated) — audits ONE module
 *   │     tools: read_file, grep, research_best_practices, consult_advisor,
 *   │            load_skill
 *   ├── research_worker (Isolated) — web_search   [consult handler: research]
 *   └── advisor         (Isolated, cloud model)   [consult handler: advice]
 * ```
 *
 * The orchestrator enumerates crates → modules, adds one `task_list` task per
 * module, dispatches the analysis worker per task, accumulates findings in
 * `memory`, finalizes the top 5, writes `workspace/findings.md`, and runs the
 * y/N issue-filing flow. The audit is READ-ONLY; the only writes are
 * `workspace/findings.md` and (approved) GitHub issues.
 *
 * ## Run it
 *
 * ```sh
 * ollama serve &
 * ollama pull gemma4:e4b
 * export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
 * pnpm start
 * ```
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  HarnessBuilder,
  OLLAMA_DEFAULT_BASE_URL,
  OllamaModelInterface,
  SessionId,
  WorkspaceScopedSandbox,
  cacheProvider,
  context as contextNs,
  newTask,
  storage,
  toolRegistry,
  type ConsultHandlerEntry,
  type ConsultHandlerMap,
  type ContextManager,
  type Harness,
  type HarnessStreamEvent,
  type HumanRequest,
  type HumanResponse,
  type PausedState,
  type RunResult,
} from "@spore/core";
import {
  StandardTools,
  SubagentTool,
  WebSearchTool,
  type ContextSharing,
} from "@spore/tools";

import { SkillCatalog, SkillInjectingContextManager } from "./skills.js";
import {
  consultAdvisorTool,
  researchBestPracticesTool,
  KIND_ADVICE,
  KIND_RESEARCH,
} from "./tools/consult.js";
import { loadSkillTool } from "./tools/load-skill.js";

const { StandardContextManager, intoHarnessAdapter, defaultCompactionConfig } =
  contextNs;
const { StorageProvider, InMemoryStorageProvider } = storage;
const { NullCacheProvider } = cacheProvider;

/** The pre-filled audit prompt the user presses enter to accept (from the
 *  issue). An empty line at the REPL ⇒ this verbatim. */
const DEFAULT_AUDIT_PROMPT =
  "Audit the current repo for the rust language. Work sequentially by " +
  "identifying each crate, and each module in the crate, and adding a task to " +
  "the tasklist for a subagent to do the deep dive audit on.";

/** Per-worker wall-clock cap. A worker can burn many internal ReAct turns (and
 *  mediated consults); this bounds how long the orchestrator waits on one
 *  delegation. */
const WORKER_TIMEOUT_MS = 300_000;

const ORCHESTRATOR_PROMPT =
  "You are the cordyceps orchestrator: an autonomous Rust-repo auditor. You do " +
  "not stop until the audit is complete. Work sequentially: (1) use `list_dir` " +
  "to enumerate the crates under `rust/crates/`, then the modules (`src/*.rs`) " +
  "in each crate; (2) for each module, add ONE task to the task list " +
  "(`task_list`) describing the module to audit; (3) for each task, call " +
  "`analysis_worker` with an `instruction` naming the ONE module to deep-dive " +
  "audit; (4) accumulate the findings each worker returns into `memory` under a " +
  "stable key; (5) when every module is audited, pick the TOP 5 most important " +
  "findings across all modules and write them as a markdown report to " +
  "`findings.md` using `write_file`. The audit is READ-ONLY — never modify " +
  "source files. Delegate the per-module deep dives to `analysis_worker`; do " +
  "not audit modules yourself. Finish by writing findings.md.";

const ANALYSIS_WORKER_PROMPT =
  "You are an analysis worker: you deep-dive audit exactly ONE Rust module for " +
  "real, actionable defects. BEFORE auditing, call `load_skill` with " +
  '`skill_id` = "audit" and follow the returned procedure and findings schema ' +
  "EXACTLY. Stay inside the one module you were given. Grep first, read only " +
  "narrow line ranges, and escalate with `research_best_practices` (idiom " +
  "questions) or `consult_advisor` (severity / is-this-real questions) when " +
  "genuinely unsure. Your FINAL answer must be a JSON array of " +
  "{file, line, severity, description} objects — and nothing else.";

const RESEARCH_PROMPT =
  "You are a research worker. Use the web_search tool to gather current, " +
  "factual information on the Rust best-practice or idiom question you are " +
  "given. Issue focused queries, read the results, and return a concise, cited " +
  "answer as plain text. Act using web_search — do not answer from memory alone.";

const ADVISOR_PROMPT =
  "You are a senior Rust advisor. A worker is stuck on whether a finding is a " +
  "real defect, or on how to rank its severity. Use `read_file` and `grep` to " +
  "investigate the specific code in question, then give a crisp, decisive " +
  "recommendation: is it a real defect, what severity " +
  "(low/medium/high/critical), and why. Be concrete and brief.";

/** The single-parameter input schema every subagent tool advertises. */
function instructionSchema(): unknown {
  return {
    type: "object",
    properties: {
      instruction: {
        type: "string",
        description: "The full instruction / task for the worker agent.",
      },
    },
    required: ["instruction"],
  };
}

/** Build the SearXNG-backed `web_search` catalogue tool (identical to 06/11). */
function buildWebSearch(endpoint: string): toolRegistry.StandardTool {
  return {
    implementation: WebSearchTool.withConfig({
      endpoint,
      method: "GET",
      queryParam: "q",
      authHeaders: [],
      bodyAuthParams: [],
    }),
    schema: WebSearchTool.schema(),
  };
}

/** Build the inner standard compaction adapter (the same one
 *  `HarnessBuilder.conversational` installs), so our skill-injecting context
 *  manager can wrap it and delegate every non-`assemble` method to it. */
function buildInnerContextManager(
  modelId: string,
  baseUrl: string,
): ContextManager {
  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const rich = new StandardContextManager(
    model,
    new NullCacheProvider(),
    defaultCompactionConfig(),
  );
  return intoHarnessAdapter(rich);
}

/** Build the research worker harness (web_search only). The `kind=research`
 *  consult handler. */
function buildResearchHarness(
  modelId: string,
  baseUrl: string,
  endpoint: string,
): Harness {
  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  return HarnessBuilder.conversational(model)
    .tool(buildWebSearch(endpoint))
    .systemPrompt(RESEARCH_PROMPT)
    .build();
}

/** Build the advisor harness (cloud model, read_file + grep). The `kind=advice`
 *  consult handler. Rides the same Ollama endpoint via `withBaseUrl`; only the
 *  model id differs. */
function buildAdvisorHarness(
  modelId: string,
  baseUrl: string,
  repoSandbox: WorkspaceScopedSandbox,
): Harness {
  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  return HarnessBuilder.conversational(model)
    .sandbox(repoSandbox)
    .tool(StandardTools.readFile())
    .tool(StandardTools.grep())
    .systemPrompt(ADVISOR_PROMPT)
    .build();
}

/** Build the analysis worker harness: read_file, grep, the two consult tools,
 *  and load_skill. Isolated; audits ONE module. It shares the SAME storage as
 *  the orchestrator so `load_skill`'s active-skill write and the context
 *  manager's read rendezvous within the run (the run store is keyed by the
 *  worker's own session id, so each worker activates `audit` for itself). */
function buildAnalysisHarness(
  modelId: string,
  baseUrl: string,
  repoSandbox: WorkspaceScopedSandbox,
  storageProvider: storage.StorageProvider,
  catalog: SkillCatalog,
): Harness {
  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const inner = buildInnerContextManager(modelId, baseUrl);
  const skillCm = new SkillInjectingContextManager(
    inner,
    storageProvider.run(),
    catalog.manifest(),
  );
  return HarnessBuilder.conversational(model)
    .sandbox(repoSandbox)
    .storage(storageProvider)
    .contextManager(skillCm)
    .tool(StandardTools.readFile())
    .tool(StandardTools.grep())
    .tool(researchBestPracticesTool())
    .tool(consultAdvisorTool())
    .tool(loadSkillTool(catalog.registry()))
    .systemPrompt(ANALYSIS_WORKER_PROMPT)
    .build();
}

/** Wrap the analysis worker as a `SubagentTool` with the consult handlers
 *  installed. The handlers mediate by `kind` (research → research_worker,
 *  advice → advisor) with the per-kind budgets + overflow policies from #114. */
function buildAnalysisTool(
  analysis: Harness,
  consultHandlers: ConsultHandlerMap,
): toolRegistry.StandardTool {
  const emptyChildRegistry = new toolRegistry.StandardToolRegistry();
  const isolated: ContextSharing = { kind: "isolated" };
  const description =
    "Delegate a deep-dive audit of ONE Rust module: pass an `instruction` " +
    "naming the module; it loads the `audit` skill, audits the module " +
    "(escalating via consults when stuck), and returns a JSON array of " +
    "{file, line, severity, description} findings.";
  const subagent = SubagentTool.buildOrThrow({
    name: "analysis_worker",
    description,
    inputSchema: instructionSchema(),
    timeoutMs: WORKER_TIMEOUT_MS,
    contextSharing: isolated,
    harness: analysis,
    childRegistry: emptyChildRegistry,
    consultHandlers,
  });

  return {
    implementation: subagent,
    schema: {
      name: "analysis_worker",
      description,
      parameters: instructionSchema(),
      // `open_world`: a subagent runs a whole agent and reaches outside the
      // process, so it is not a closed, read-only computation.
      annotations: {
        read_only: false,
        destructive: false,
        idempotent: false,
        open_world: true,
      },
    },
  };
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const modelId =
    argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "gemma4:e4b";
  const advisorModelId =
    argValue(args, "--advisor-model") ??
    process.env.SPORE_ADVISOR_MODEL ??
    "minimax-m3:cloud";
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;

  // Required search backend (research worker) — fail fast like 06/11.
  const endpoint =
    argValue(args, "--search-url") ??
    process.env.SPORE_WEB_SEARCH_ENDPOINT?.trim();
  if (!endpoint) {
    console.error(
      "SPORE_WEB_SEARCH_ENDPOINT is not set.\n" +
        "Set it to a SearXNG JSON endpoint, e.g.\n" +
        '  export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"\n' +
        "See .env.example and the README.",
    );
    process.exit(2);
  }

  // Resolve the repo root (cwd) for the read-only audit sandbox, and this
  // example's workspace/ for the report write.
  const repoRoot = process.cwd();
  const here = dirname(fileURLToPath(import.meta.url));
  const workspaceRoot = join(here, "..", "workspace");
  mkdirSync(workspaceRoot, { recursive: true });

  // The bundled `audit` skill, read from disk (the TS analogue of Rust's
  // `include_str!`). Seed `.spore/skills/audit/SKILL.md` from it if absent so a
  // user can see the filesystem-registry shape (documented in the README).
  const bundledAudit = readFileSync(
    join(here, "..", "skills", "audit", "SKILL.md"),
    "utf8",
  );
  seedBundledAuditSkill(repoRoot, bundledAudit);

  // One in-memory storage provider, shared by the orchestrator and the analysis
  // worker so `load_skill` (worker-side write) and the context manager (read)
  // rendezvous on `runStore["active_skills"]`.
  const storageProvider = StorageProvider.single(new InMemoryStorageProvider());

  // Scan + register skills (bundled audit + any project/user skills).
  const catalog = await SkillCatalog.bootstrap(repoRoot, bundledAudit);

  // The orchestrator can read the whole repo + write findings.md into its own
  // workspace and run `gh` via bash_command. Workers/advisor get the same root
  // but only read_file/grep, so their reads stay inside the repo. We rely on the
  // prompt + skill discipline for the read-only guarantee (the orchestrator must
  // also write findings.md), not a read-only sandbox.
  const orchestratorSandbox = new WorkspaceScopedSandbox({ root: repoRoot });
  const workerSandbox = new WorkspaceScopedSandbox({ root: repoRoot });

  // ---- Build the consult handlers (research + advice) -----------------------
  const researchHandler = buildResearchHarness(modelId, baseUrl, endpoint);
  const advisorHandler = buildAdvisorHarness(
    advisorModelId,
    baseUrl,
    workerSandbox,
  );
  const consultHandlers: ConsultHandlerMap = new Map<
    string,
    ConsultHandlerEntry
  >([
    [
      KIND_RESEARCH,
      {
        handler: researchHandler,
        budget: 5,
        overflow: { kind: "soft_fail" },
      },
    ],
    [
      KIND_ADVICE,
      {
        handler: advisorHandler,
        budget: 3,
        overflow: { kind: "escalate_to_human" },
      },
    ],
  ]);

  // ---- Build the analysis worker + wrap it (with consult handlers) ----------
  const analysis = buildAnalysisHarness(
    modelId,
    baseUrl,
    workerSandbox,
    storageProvider,
    catalog,
  );
  const analysisTool = buildAnalysisTool(analysis, consultHandlers);

  // ---- Build the orchestrator -----------------------------------------------
  const orchestratorModel = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const orchestrator = HarnessBuilder.conversational(orchestratorModel)
    .sandbox(orchestratorSandbox)
    .storage(storageProvider)
    .tool(StandardTools.listDir())
    .tool(StandardTools.grep())
    .tool(StandardTools.taskList())
    .tool(StandardTools.memory())
    .tool(StandardTools.writeFile())
    .tool(StandardTools.bashCommand())
    .tool(analysisTool)
    .systemPrompt(ORCHESTRATOR_PROMPT)
    .build();

  console.log(`model        : ${modelId}`);
  console.log(`advisor model: ${advisorModelId}`);
  console.log(`endpoint     : ${endpoint}`);
  console.log(`repo root    : ${repoRoot}`);
  console.log(`workspace    : ${workspaceRoot}`);
  console.log(
    `skills       : ${catalog
      .manifest()
      .map((e) => e.name)
      .join(", ")}`,
  );
  console.log("strategy     : orchestrator=ReAct, workers=ReAct (isolated)\n");

  // ---- REPL: pre-filled prompt, enter accepts the default -------------------
  const prompt = await readAuditPrompt();

  // Stream banners: mirror 11-multi-agent's `┌─ … └─` boundary style for the
  // analysis_worker delegation.
  const callNames = new Map<string, string>();
  const onStream = makeStreamSink(callNames);

  const task = newTask(
    prompt,
    SessionId.generate(),
    { kind: "re_act", max_iterations: 64 },
    { max_turns: 64 },
  );

  // ---- Drive the orchestrator, handling the human-escalation ladder ---------
  let result: RunResult;
  try {
    result = await orchestrator.run({ task, on_stream: onStream });
  } catch (err) {
    console.error(
      `\ncould not reach the model — is Ollama running at ${baseUrl}? (\`ollama serve\`)\n${err instanceof Error ? err.message : String(err)}`,
    );
    process.exit(1);
  }

  for (;;) {
    if (result.kind === "success") {
      console.log(
        `\norchestrator done (${result.turns} turn(s)): ${truncate(result.output, 400)}`,
      );
      const findings = join(workspaceRoot, "findings.md");
      if (existsSync(findings)) {
        console.log(`\nfindings.md written: ${findings}`);
        await runIssueFilingFlow(orchestrator, findings);
      } else {
        console.error(
          "\nwarning: orchestrator finished but workspace/findings.md was not written.",
        );
      }
      return;
    }
    if (result.kind === "failure") {
      console.error(
        `\norchestrator failed after ${result.turns} turn(s): ${JSON.stringify(result.reason)}`,
      );
      process.exit(1);
    }
    if (result.kind === "waiting_for_human") {
      // The advice consult budget (3) was exhausted under escalate_to_human: the
      // analysis_worker SubagentTool converted the over-budget consult into a
      // human pause, which bubbled up here.
      result = await handleHumanEscalation(
        orchestrator,
        advisorHandler,
        result.state,
        result.request,
      );
      continue;
    }
    console.error(`\nrun ended unexpectedly: ${JSON.stringify(result)}`);
    process.exit(1);
  }
}

/** Build the orchestrator stream sink: boundary banners for the analysis worker,
 *  terse lines for the standard tools (mirrors 11-multi-agent). */
function makeStreamSink(
  callNames: Map<string, string>,
): (event: HarnessStreamEvent) => void {
  return (event: HarnessStreamEvent): void => {
    switch (event.kind) {
      case "turn_start":
        console.log(`orchestrator · turn ${event.turn}`);
        break;
      case "tool_call": {
        callNames.set(event.call_id, event.name);
        if (event.name === "analysis_worker") {
          const instruction = readInstruction(event.args);
          console.log("┌─ orchestrator → analysis_worker");
          console.log(`│  received: ${truncate(instruction, 200)}`);
        } else {
          console.log(
            `  orchestrator → ${event.name}(${truncate(JSON.stringify(event.args ?? {}), 140)})`,
          );
        }
        break;
      }
      case "tool_result": {
        const name = callNames.get(event.call_id) ?? "<tool>";
        callNames.delete(event.call_id);
        const content = event.content ?? "";
        if (name === "analysis_worker") {
          const tag = event.is_error ? "FAILED" : "findings";
          console.log("└─ analysis_worker → orchestrator");
          console.log(`   ${tag}: ${truncate(content, 300)}`);
        } else {
          const tag = event.is_error ? "err" : "ok";
          console.log(
            `  ${name} → orchestrator [${tag}]: ${truncate(content, 140)}`,
          );
        }
        break;
      }
      default:
        break;
    }
  };
}

/**
 * Handle an advice-budget-exhausted human escalation with the three-choice
 * ladder. Returns the next `RunResult` (the orchestrator resumed).
 *
 * IMPORTANT (honest mechanics): the worker's paused consult lives inside the
 * orchestrator's `PausedState.child_state`, and the harness does NOT yet wire a
 * child-consult resume through the parent. So every choice here resumes the
 * ORCHESTRATOR with the human's decision injected as guidance; the specific
 * module's in-flight worker audit is dropped. "+1 advisor turn" re-runs the
 * advisor handler HOST-SIDE and injects its answer as that guidance — the
 * closest we can get to a budget bump without a core primitive. (See the README;
 * the lossless child-resume is the #5/#115 follow-up.)
 */
async function handleHumanEscalation(
  orchestrator: Harness,
  advisorHandler: Harness,
  state: PausedState,
  request: HumanRequest,
): Promise<RunResult> {
  const context = humanRequestText(request);
  console.log("\n╔═ HUMAN ESCALATION (advisor budget exhausted) ═");
  console.log(`║ ${truncate(context, 400)}`);
  console.log("╚═══════════════════════════════════════════════");
  console.log(
    "Choose: [1] +1 advisor turn  [2] abort subagent & chat  [3] free-form answer",
  );

  const choice = (await promptLine("> ")).trim();
  if (choice === "1") {
    console.log("(running advisor for one more turn…)");
    const advisorTask = newTask(context, SessionId.generate(), {
      kind: "re_act",
      max_iterations: 16,
    });
    const r = await advisorHandler.run({ task: advisorTask });
    const advice =
      r.kind === "success"
        ? r.output
        : `advisor did not complete cleanly: ${JSON.stringify(r)}`;
    console.log(`advisor: ${truncate(advice, 300)}`);
    const response: HumanResponse = { kind: "answer", text: advice };
    return orchestrator.resume(state, response);
  }
  if (choice === "2") {
    console.log(
      "(aborting the stuck subagent; returning to the orchestrator…)",
    );
    const response: HumanResponse = { kind: "halt" };
    return orchestrator.resume(state, response);
  }
  const text = choice === "3" ? await promptLine("your answer> ") : choice;
  const response: HumanResponse = { kind: "answer", text };
  return orchestrator.resume(state, response);
}

/** After a successful audit, present the top-5 and offer to file them as issues.
 *  The model drives `gh issue create` via `bash_command` (no `gh` skill). */
async function runIssueFilingFlow(
  orchestrator: Harness,
  findings: string,
): Promise<void> {
  let report = "";
  try {
    report = readFileSync(findings, "utf8");
  } catch {
    report = "";
  }
  console.log("\n── top findings (workspace/findings.md) ──");
  console.log(truncate(report, 1200));
  console.log("──────────────────────────────────────────");

  const answer = await promptLine("File these as GitHub issues? [y/N] ");
  if (answer.trim().toLowerCase() !== "y") {
    console.log("Not filing. Done.");
    return;
  }

  console.log(
    "(asking the orchestrator to file the top 5 via `gh issue create`…)",
  );
  const task = newTask(
    "Using `bash_command`, file the TOP 5 findings from workspace/findings.md " +
      "as GitHub issues via `gh issue create` — one issue per finding, with a " +
      "clear title and a body containing the file, line, severity, and " +
      "description. Run `gh` once per finding. Report the issue URLs when done.",
    SessionId.generate(),
    { kind: "re_act", max_iterations: 24 },
  );
  const result = await orchestrator.run({ task });
  if (result.kind === "success") {
    console.log(`\nfiling done: ${truncate(result.output, 400)}`);
  } else {
    console.error(
      `\nfiling did not complete cleanly: ${JSON.stringify(result)}`,
    );
  }
}

/** Seed `.spore/skills/audit/SKILL.md` from the bundled copy if absent. */
function seedBundledAuditSkill(repoRoot: string, bundledAudit: string): void {
  const dir = join(repoRoot, ".spore", "skills", "audit");
  const file = join(dir, "SKILL.md");
  if (existsSync(file)) return;
  try {
    mkdirSync(dir, { recursive: true });
    writeFileSync(file, bundledAudit);
  } catch {
    // best-effort: a read-only cwd just means no seed file (the bundled skill is
    // still registered directly in SkillCatalog.bootstrap).
  }
}

/** Read the audit prompt from the REPL: print the default, accept an empty line
 *  as the default verbatim. */
async function readAuditPrompt(): Promise<string> {
  console.log(
    "Default audit prompt (press enter to accept, or type your own):",
  );
  console.log(`  ${DEFAULT_AUDIT_PROMPT}`);
  const line = await promptLine("audit> ");
  return line.trim() === "" ? DEFAULT_AUDIT_PROMPT : line;
}

/** Print a prompt and read one line from stdin (trailing newline stripped). */
async function promptLine(prompt: string): Promise<string> {
  const rl = createInterface({ input, output });
  try {
    const answer = await rl.question(prompt);
    return answer.replace(/[\r\n]+$/, "");
  } catch {
    return "";
  } finally {
    rl.close();
  }
}

/** Extract the human-facing text from a {@link HumanRequest}. */
function humanRequestText(request: HumanRequest): string {
  switch (request.kind) {
    case "review":
      return request.content;
    case "clarification":
      return request.question;
    case "tool_approval":
      return "tool approval requested";
  }
}

/** Pull the `instruction` string out of a `tool_call`'s args, if present. */
function readInstruction(args: unknown): string {
  if (
    typeof args === "object" &&
    args !== null &&
    typeof (args as Record<string, unknown>).instruction === "string"
  ) {
    return (args as Record<string, unknown>).instruction as string;
  }
  return "<no instruction>";
}

function argValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

/** Keep boundary lines readable. */
function truncate(s: string, max: number): string {
  const flat = s.replace(/\n/g, " ");
  const chars = [...flat];
  return chars.length <= max ? flat : `${chars.slice(0, max).join("")}…`;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
