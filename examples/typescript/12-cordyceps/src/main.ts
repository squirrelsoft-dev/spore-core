/**
 * spore-core example 12 — **cordyceps**: the capstone of the Composable
 * Execution refactor (#117–#131).
 *
 * **The thesis: you describe a strategy as DATA — a composed `LoopStrategy`
 * tree — wire its string handles to concrete collaborators in an
 * {@link ExecutionRegistry}, and the harness runs the whole nested machine under
 * one shared budget / usage / observability context.**
 *
 * The motivating composition is:
 *
 * ```text
 * Ralph[ PlanExecute[ ReAct, SelfVerifying[ ReAct ] ] ]
 * │       │             │      │             │
 * │       │             │      │             └─ worker: audits ONE module
 * │       │             │      └─ Default-FAIL evaluator (single read-only turn)
 * │       │             └─ plan: explores the repo, builds a blocker-aware DAG
 * │       └─ plan→ready-set: walks the DAG in dependency order, self-verifying each task
 * └─ continuation wrapper: resets the window, resumes from durable progress
 * ```
 *
 * ## What changed vs. the pre-#131 example (HONEST note)
 *
 * The old depth-1 example used a hand-built `SubagentTool` orchestrator with a
 * per-node consult mediator (#114) and an architect-side `load_skill` tool
 * (#115). The declarative tree has NO SubagentTool seam, so:
 *
 * - the #114 consult ladder is **PRESERVED, with its mediation seam moved**. The
 *   worker still calls `research_best_practices` / `consult_advisor`, which lower
 *   to `ToolOutput` `consult`. With no `SubagentTool` to mediate, the worker-leaf
 *   consult propagates all the way up to a top-level {@link RunResult} `consult`,
 *   and the HOST run loop mediates it — routing by `kind` to a helper harness
 *   with a per-kind budget + overflow policy (`research` → web_search, budget 5,
 *   soft_fail; `advice` → cloud advisor, budget 3, escalate_to_human). Identical
 *   #114 semantics, host-owned budgets.
 * - `load_skill` is **dropped** — there is no worker-side per-node seam;
 * - the `audit` skill is **kept**, but now rides the single GLOBAL
 *   {@link SkillInjectingContextManager} (the harness's `context_manager`),
 *   seeded ALWAYS-ACTIVE at startup. The audit procedure reaches the model
 *   structurally every turn, compaction-proof, with no `load_skill` round-trip.
 *
 * ## The tree is DATA
 *
 * We do NOT hand-build the {@link LoopStrategy}. We read the canonical fixture
 * `fixtures/strategy/cordyceps_tree.json` and deserialize it — so this example
 * proves the canonical fixture deserializes and runs.
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

import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { mkdirSync, readFileSync, realpathSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  AgentId,
  EmptyToolRegistry,
  ExecutionRegistry,
  HarnessBuilder,
  LoopStrategySchema,
  ModelAgent,
  OLLAMA_DEFAULT_BASE_URL,
  OllamaModelInterface,
  SessionId,
  StandardHarness,
  WorkspaceScopedSandbox,
  cacheProvider,
  context as contextNs,
  loopStrategyMaxSteps,
  newTask,
  reactPerLoop,
  storage,
  toolRegistry,
  verifier,
  type Agent,
  type ConsultHandlerEntry,
  type ConsultRequest,
  type ContextManager,
  type EscalationAction,
  type EscalationMode,
  type Harness,
  type HumanRequest,
  type HumanResponse,
  type LoopStrategy,
  type PausedState,
  type RunResult,
  type Task,
} from "@spore/core";
import { StandardTools, WebSearchTool } from "@spore/tools";

import {
  SkillCatalog,
  SkillInjectingContextManager,
  ACTIVE_SKILLS_KEY,
} from "./skills.js";
import {
  consultAdvisorTool,
  researchBestPracticesTool,
  KIND_ADVICE,
  KIND_RESEARCH,
} from "./tools/consult.js";
import { sendUserMessageTool } from "./tools/send-message.js";

const { StandardContextManager, intoHarnessAdapter, defaultCompactionConfig } =
  contextNs;
const { ProjectId, CompositeStorageProvider, FileSystemStorageProvider } =
  storage;
const { NullCacheProvider } = cacheProvider;
const { EvaluatorResponseVerifier } = verifier;

const here = dirname(fileURLToPath(import.meta.url));

/** The canonical composed-strategy fixture, read so the example proves the
 *  ground-truth tree deserializes (and runs) verbatim — never hand-built. */
const CORDYCEPS_TREE_PATH = resolve(
  here,
  "..",
  "..",
  "..",
  "..",
  "fixtures",
  "strategy",
  "cordyceps_tree.json",
);

/** Bundled `audit` skill (the global context_manager's always-active procedure). */
const BUNDLED_AUDIT_PATH = join(here, "..", "skills", "audit", "SKILL.md");

/** The verifier registry key the `SelfVerifying` node's `evaluator` resolves to. */
export const EXEC_EVALUATOR_KEY = "exec-evaluator";

/** The pre-filled audit prompt (press enter to accept). */
const DEFAULT_AUDIT_PROMPT =
  "Audit this repository for Rust defects. Discover the crates and their modules, " +
  "audit each module for real, actionable defects, and write a markdown report of " +
  "the most important findings to `workspace/findings.md`.";

const EXEC_SYSTEM_PROMPT =
  "You are a cordyceps execution machine. Your strategy is composed " +
  "declaratively: a Ralph continuation wrapper drives a PlanExecute, whose plan " +
  "phase explores the repo and builds a blocker-aware task graph via `task_list`, " +
  "and whose execute phase walks that graph as a ready-set — auditing one module " +
  "per ready task, each result self-verified by a read-only evaluator " +
  "(Default-FAIL: only an explicit PASS clears a task).\n\n" +
  "Before each step, call `send_user_message` with one short sentence telling the " +
  "watching human what you are about to do and why.\n\n" +
  "You are already scoped to the repository root (READ-ONLY). Use `.` for the root " +
  "and paths relative to it (e.g. `rust/crates`); never prefix a path with the " +
  "repository's own folder name. The audit is read-only — you have no write tool; " +
  "never attempt to modify source files.\n\n" +
  "Follow the ACTIVE `audit` skill's procedure and output schema exactly: grep " +
  "first, read narrow, and return findings as a JSON array of " +
  "{file, line, severity, description}.\n\n" +
  "PLAN phase: explore the repo with `list_dir`/`grep`, then build a blocker-aware " +
  "task graph with `task_list` (one task per module; add dependencies where one " +
  "audit should wait on another). RALPH wrapper: resume from durable `task_list` " +
  "progress after each context-window reset and keep going until every task is done.";

const RESEARCH_PROMPT =
  "You are a research worker. A peer agent needs factual, current information on a " +
  "Rust best-practice or language question.\n\n" +
  "Use `web_search` to find the answer. Issue focused queries, read the results, " +
  "and return a concise cited answer in plain text. Do not answer from memory " +
  "alone — always search first.";

const ADVISOR_PROMPT =
  "You are a senior Rust advisor. A worker has escalated a candidate finding to you " +
  "because they need a judgment call.\n\n" +
  "Use `read_file` and `grep` to examine the specific code in question. Then make a " +
  "decision: is this a real defect, what is the severity " +
  "(low / medium / high / critical), and why. Be decisive. State your verdict in " +
  "one sentence, your reasoning in two. Do not hedge.";

/** `plan-schema` — the task-graph contract the plan phase's ReAct emits. */
function planSchema(): unknown {
  return {
    type: "object",
    properties: {
      tasks: {
        type: "array",
        description:
          "Ordered task-graph entries; each names a module to audit.",
        items: {
          type: "object",
          properties: {
            module: { type: "string", description: "Module path to audit." },
            blockers: {
              type: "array",
              items: { type: "integer" },
              description: "1-based ids of tasks this one waits on.",
            },
          },
          required: ["module"],
        },
      },
      rationale: { type: "string" },
    },
    required: ["tasks"],
  };
}

/** `worker-schema` — the per-module finding contract the worker ReAct emits. */
function workerSchema(): unknown {
  return {
    type: "array",
    description: "Findings for ONE module.",
    items: {
      type: "object",
      properties: {
        file: {
          type: "string",
          description: "Path relative to the repo root.",
        },
        line: { type: "integer", description: "1-based line of the defect." },
        severity: { enum: ["low", "medium", "high", "critical"] },
        description: {
          type: "string",
          description: "Concrete, actionable defect.",
        },
      },
      required: ["file", "line", "severity", "description"],
    },
  };
}

/** The `plan-tools` catalogue: explore + author the task graph (read-only). */
function planTools(): toolRegistry.StandardTool[] {
  return [
    StandardTools.listDir(),
    StandardTools.grep(),
    StandardTools.taskList(),
  ];
}

/** The `exec-tools` catalogue: read-only audit + the #114 consult ladder + human
 *  observability. The two consult tools lower to `ToolOutput` `consult`, which the
 *  host run loop mediates (the seam moved off `SubagentTool`). */
function execTools(): toolRegistry.StandardTool[] {
  return [
    StandardTools.readFile(),
    StandardTools.grep(),
    researchBestPracticesTool(),
    consultAdvisorTool(),
    sendUserMessageTool("🤖"),
  ];
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

/** Build the research handler harness (web_search only) — the `kind="research"`
 *  consult handler. Run host-side on a `ConsultRequest`. */
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

/** Build the advisor handler harness (cloud model, read_file + grep) — the
 *  `kind="advice"` consult handler. Rides the same Ollama endpoint via
 *  `withBaseUrl`; only the model id differs (heterogeneous models). */
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

/** Build the HOST-owned `kind → {handler, budget, overflow}` map (#114). The
 *  composed tree has no `SubagentTool`, so the host run loop holds these entries
 *  and mediates each `RunResult` `consult` against them — the per-kind budget
 *  lives for the whole run (see `mediateConsult`). */
function buildConsultHandlers(
  research: Harness,
  advisor: Harness,
): Map<string, ConsultHandlerEntry> {
  return new Map<string, ConsultHandlerEntry>([
    [
      KIND_RESEARCH,
      { handler: research, budget: 5, overflow: { kind: "soft_fail" } },
    ],
    [
      KIND_ADVICE,
      { handler: advisor, budget: 3, overflow: { kind: "escalate_to_human" } },
    ],
  ]);
}

/** Build a model agent over the local Ollama model. */
function modelAgent(id: string, modelId: string, baseUrl: string): Agent {
  return new ModelAgent(
    AgentId.of(id),
    OllamaModelInterface.withBaseUrl(modelId, baseUrl),
  );
}

/** The Default-FAIL self-verification evaluator registered under `exec-evaluator`.
 *  A single read-only turn (`max_iterations = 1`); the neither-pattern ⇒ Failed
 *  contract is built into {@link verifier.EvaluatorResponseVerifier}. */
export function execEvaluator(): verifier.Verifier {
  return new EvaluatorResponseVerifier({
    pass_pattern: "(?i)\\bPASS\\b",
    fail_pattern: "(?i)\\bFAIL\\b",
    max_iterations: 1,
  });
}

/** Assemble the {@link ExecutionRegistry} the cordyceps tree's handles resolve
 *  against: agents `planner`/`executor`/`ralph-agent`, toolsets
 *  `plan-tools`/`exec-tools`, schemas `plan-schema`/`worker-schema`, and the
 *  `exec-evaluator` verifier. The handle STRINGS are ground truth from the
 *  fixture; this is the host-side wiring of those strings to collaborators. */
export function buildRegistry(
  modelId: string,
  baseUrl: string,
): ExecutionRegistry {
  return (
    ExecutionRegistry.builder()
      .agent("planner", modelAgent("planner", modelId, baseUrl))
      .agent("executor", modelAgent("executor", modelId, baseUrl))
      .agent("ralph-agent", modelAgent("ralph-agent", modelId, baseUrl))
      // The toolset HANDLES must resolve for `validate()`. Per-node scoping is now
      // RESOLVED (Issue 2): each node dispatches its OWN toolset catalogue, wired
      // per-key on the HarnessBuilder (`.toolsetTools("plan-tools", ...)` /
      // `.toolsetTools("exec-tools", ...)`, see `main`). These registry slots are
      // validation-only presence entries — never dispatched — so an
      // `EmptyToolRegistry` placeholder suffices. (`buildConfig` also auto-fills
      // these same presence entries from `.toolsetTools`; keeping the explicit
      // entries here makes the standalone registry `validate()` contract
      // self-consistent without the builder.)
      .toolset("plan-tools", new EmptyToolRegistry())
      .toolset("exec-tools", new EmptyToolRegistry())
      .schema("plan-schema", planSchema())
      .schema("worker-schema", workerSchema())
      .verifier(EXEC_EVALUATOR_KEY, execEvaluator())
      .build()
  );
}

/** Build the inner standard compaction adapter (the same one
 *  `HarnessBuilder.conversational` installs), so the skill-injecting context
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

/** Build the GLOBAL skill-injecting context manager with the `audit` skill seeded
 *  ALWAYS-ACTIVE for `session`. Wraps the standard compaction adapter. */
async function buildGlobalContextManager(
  modelId: string,
  baseUrl: string,
  storageProvider: storage.StorageProvider,
  repoRoot: string,
  session: SessionId,
): Promise<ContextManager> {
  const bundledAudit = readFileSync(BUNDLED_AUDIT_PATH, "utf8");
  const catalog = await SkillCatalog.bootstrap(repoRoot, bundledAudit);
  // Seed `audit` always-active: the global context_manager injects its body
  // structurally every turn (no `load_skill` round-trip in the composed tree).
  await storageProvider.run().put(session, ACTIVE_SKILLS_KEY, ["audit"]);

  const inner = buildInnerContextManager(modelId, baseUrl);
  return new SkillInjectingContextManager(
    inner,
    storageProvider.run(),
    catalog.manifest(),
  );
}

/** Build the cordyceps {@link Task}: the composed tree deserialized from the
 *  fixture, under a generous global backstop so the per-node `per_loop{12}`
 *  worker bound fires first. */
export function buildTask(prompt: string, session: SessionId): Task {
  const tree = readCordycepsTree();
  return newTask(prompt, session, tree, { max_turns: 64 });
}

/** Read + deserialize the canonical cordyceps tree from the shared fixture. */
function readCordycepsTree(): LoopStrategy {
  return LoopStrategySchema.parse(
    JSON.parse(readFileSync(CORDYCEPS_TREE_PATH, "utf8")),
  );
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const modelId =
    argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "gemma4:e4b";
  // The #114 advisor consult handler runs a heterogeneous (cloud) model.
  const advisorModelId =
    argValue(args, "--advisor-model") ??
    process.env.SPORE_ADVISOR_MODEL ??
    "minimax-m3:cloud";
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;

  // The research consult handler needs a SearXNG JSON endpoint — fail fast (like
  // examples 06/11) so a missing backend is a startup error, not a mid-run
  // surprise.
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

  // #142: canonicalize the repo root FIRST (resolves symlinks / relative
  // components / macOS case) so the derived project id is stable.
  const repoRoot = realpathSync(process.cwd());
  const workspaceRoot = join(repoRoot, "workspace");
  mkdirSync(workspaceRoot, { recursive: true });

  // #142: a STABLE project id derived from the canonicalized repo root (decision
  // 5 — derived from the workspace root, NOT process cwd). It keys the DURABLE
  // task_list / plan / Ralph checkpoint so they survive Ralph window resets AND
  // process restarts. `repoRoot` is already canonicalized above, so the pure
  // derivation is exact here.
  const projectId = ProjectId.fromCanonicalPath(repoRoot);
  // The CENTRAL durable root, à la Claude Code: `~/.spore/projects/<project_id>/`
  // (decision 1). Tests/CI must point HOME at a tempdir — this never writes to
  // the real `~/.spore` under test.
  const sporeRoot = join(homedir(), ".spore", "projects", projectId.asString());
  mkdirSync(sporeRoot, { recursive: true });

  // AC5: the fully-bounded tree's worst-case per-window turn count is computable
  // BEFORE the run. Ralph[PlanExecute[ReAct{4}, SelfVerifying[ReAct{12}]]] =
  // 4 + (12 + 1) = 17. An `unlimited` anywhere would collapse this to undefined.
  const treePreview = readCordycepsTree();
  console.log(`model        : ${modelId}`);
  console.log(`advisor model: ${advisorModelId}`);
  console.log(`search       : ${endpoint}`);
  console.log(`repo root    : ${repoRoot}`);
  console.log(`project id   : ${projectId.asString()}`);
  console.log(`durable root : ${sporeRoot}`);
  console.log(
    "strategy     : Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]] (from fixture)",
  );
  console.log(
    `max_steps    : ${loopStrategyMaxSteps(treePreview)}  ` +
      "(per-window worst case; unlimited anywhere ⇒ undefined)",
  );
  console.log(
    "consults     : research(web_search, budget 5, soft-fail), advice(advisor, budget 3, escalate)",
  );
  console.log();

  for (;;) {
    const prompt = await readAuditPrompt();
    if (prompt === undefined) break; // EOF (Ctrl-D) quits the REPL.

    const session = SessionId.generate();
    // #142: route the DURABLE run domain to a FileSystemStorageProvider under the
    // central root (atomic write-rename) so the task_list / plan / Ralph
    // checkpoint persist across context-window resets AND process restarts.
    // Session and observability share the same durable root; memory stays
    // project-scoped. The InMemory wiring (RAM-only, lost on crash) is gone.
    const durable = new FileSystemStorageProvider(sporeRoot);
    const storageProvider = new CompositeStorageProvider()
      .run(durable)
      .session(durable)
      .observability(durable)
      .memory("project", durable)
      .build();

    // Read-only repo sandbox: the audit never writes source files.
    const sandbox = new WorkspaceScopedSandbox({
      root: repoRoot,
      read_only: true,
    });
    // The advisor handler gets its own read-only view of the repo (read_file +
    // grep) so it can inspect the code the worker is asking about.
    const advisorSandbox = new WorkspaceScopedSandbox({
      root: repoRoot,
      read_only: true,
    });

    // The HOST-owned consult ladder (#114). The seam moved off `SubagentTool`:
    // the host loop holds these handlers + per-kind budgets for the whole run.
    const consultHandlers = buildConsultHandlers(
      buildResearchHarness(modelId, baseUrl, endpoint),
      buildAdvisorHarness(advisorModelId, baseUrl, advisorSandbox),
    );

    const registry = buildRegistry(modelId, baseUrl);
    const contextManager = await buildGlobalContextManager(
      modelId,
      baseUrl,
      storageProvider,
      repoRoot,
      session,
    );

    // The harness's own model drives the Ralph wrapper; the per-node agents come
    // from the registry. Compaction/summarization uses this model too.
    const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
    const escalationMode: EscalationMode = { kind: "surface_to_human" };
    // Issue 2 (per-node toolset scoping): each node dispatches ONLY its own
    // toolset. The planner leaf (handle `plan-tools`) sees list_dir/grep/task_list;
    // the executor leaf (handle `exec-tools`) sees read_file/grep/research/consult/
    // send_message. The union no longer leaks across nodes — the planner can no
    // longer call exec-only tools, nor the executor plan-only tools.
    const harness = HarnessBuilder.conversational(model)
      .sandbox(sandbox)
      .storage(storageProvider)
      // #142: pin the stable project id so durable artifacts key by it (rather
      // than the per-window SessionId.generate() above).
      .projectId(projectId)
      .registry(registry)
      .escalationMode(escalationMode)
      .systemPrompt(EXEC_SYSTEM_PROMPT)
      .contextManager(contextManager)
      .toolsetTools("plan-tools", planTools())
      .toolsetTools("exec-tools", execTools())
      .build();

    const task = buildTask(prompt, session);
    // Per-RUN consult counts (host-owned, #114): how many consults of each `kind`
    // have already been mediated. Persists across every pause/resume of THIS audit
    // so the per-kind budget bounds the whole run, not one turn.
    const consultCounts = new Map<string, number>();

    let result: RunResult;
    try {
      result = await harness.run({ task });
    } catch (err) {
      console.error(
        `\ncould not reach the model — is Ollama running at ${baseUrl}? (\`ollama serve\`)\n` +
          `${err instanceof Error ? err.message : String(err)}`,
      );
      process.exit(1);
    }

    for (;;) {
      if (result.kind === "success") {
        console.log(
          `\ndone (${result.turns} turn(s)): ${truncate(result.output, 400)}`,
        );
        break;
      }
      if (result.kind === "failure") {
        console.error(
          `\nfailed after ${result.turns} turn(s): ${JSON.stringify(result.reason)}`,
        );
        break;
      }
      if (result.kind === "consult") {
        // A worker leaf consult propagated up through the composed tree (no
        // SubagentTool to absorb it). The host mediates.
        result = await mediateConsult(
          harness,
          consultHandlers,
          consultCounts,
          result.request,
          result.state,
        );
        continue;
      }
      if (result.kind === "waiting_for_human") {
        result = await handleHumanEscalation(
          harness,
          result.state,
          result.request,
        );
        continue;
      }
      console.error(`\nrun ended unexpectedly: ${JSON.stringify(result)}`);
      break;
    }
  }

  console.log("\nbye.");
}

/**
 * Mediate one worker leaf consult HOST-SIDE (#114, seam relocated off
 * `SubagentTool`). Routes by `kind`, enforces the per-kind budget held in
 * `counts` for the whole run, runs the handler harness as a direct child, and
 * resumes the paused composed tree with the answer — or applies the overflow
 * policy (`soft_fail` resumes with `budget_exhausted`; `escalate_to_human`
 * surfaces the advisor ladder to the operator). Identical to the old
 * `SubagentTool.mediateConsult`, only the owner moved.
 */
async function mediateConsult(
  harness: StandardHarness,
  handlers: Map<string, ConsultHandlerEntry>,
  counts: Map<string, number>,
  request: ConsultRequest,
  state: PausedState,
): Promise<RunResult> {
  // No handler for this kind ⇒ resume the worker without help (loud, not a silent
  // stall). Matches SubagentTool R6 graceful degradation in spirit.
  const entry = handlers.get(request.kind);
  if (!entry) {
    console.error(
      `\n(no consult handler for kind ${JSON.stringify(request.kind)}; worker proceeds)`,
    );
    return harness.resumeConsult(state, {
      kind: "budget_exhausted",
      message: `no consult handler for kind ${JSON.stringify(request.kind)}; proceed without further help`,
    });
  }

  // Per-kind budget: `used` is how many consults of this kind were already
  // mediated this run. The handler runs while `used < budget`; the (budget+1)th
  // consult overflows.
  const used = counts.get(request.kind) ?? 0;
  if (used >= entry.budget) {
    if (entry.overflow.kind === "soft_fail") {
      console.log(
        `\n(consult budget for ${JSON.stringify(request.kind)} exhausted — worker finishes with what it has)`,
      );
      return harness.resumeConsult(state, {
        kind: "budget_exhausted",
        message: `consult budget for kind ${JSON.stringify(request.kind)} exhausted; proceed without further help`,
      });
    }
    return handleConsultOverflow(harness, entry, request, state);
  }

  // Run the handler harness as a direct child (depth-1, never under the worker)
  // on the consult rendered to text, then resume with its answer.
  counts.set(request.kind, used + 1);
  console.log(
    `\n┌─ consult (${request.kind}) → ${used + 1} of ${entry.budget} budget`,
  );
  const childTask = newTask(
    renderConsultInstruction(request),
    SessionId.generate(),
    reactPerLoop(16),
  );
  const r = await entry.handler.run({ task: childTask });
  // A handler that does not cleanly complete must not stall the worker — feed its
  // failure text back as the answer so the worker can adapt.
  const answer =
    r.kind === "success"
      ? r.output
      : `consult handler did not complete cleanly: ${JSON.stringify(r)}`;
  console.log(`└─ consult answer: ${truncate(answer, 200)}`);
  return harness.resumeConsult(state, { kind: "answer", text: answer });
}

/** Render a {@link ConsultRequest} to the handler's instruction text (#114). */
function renderConsultInstruction(request: ConsultRequest): string {
  return (
    `A worker agent is requesting help (kind: ${request.kind}).\n\n` +
    `Situation: ${request.situation}\n\n` +
    `Attempts so far: ${request.attempts}\n\n` +
    `Question: ${request.question}`
  );
}

/**
 * The `advice` consult overflowed its budget under `escalate_to_human`: present
 * the #114 three-choice ladder to the operator and resume the worker with the
 * decision. Preserves the original ladder semantics host-side.
 */
async function handleConsultOverflow(
  harness: StandardHarness,
  entry: ConsultHandlerEntry,
  request: ConsultRequest,
  state: PausedState,
): Promise<RunResult> {
  console.log("\n╔═ HUMAN ESCALATION (advisor budget exhausted) ═");
  console.log(`║ situation: ${truncate(request.situation, 200)}`);
  console.log(`║ question : ${truncate(request.question, 200)}`);
  console.log("║ [1] run the advisor once more (host-side)");
  console.log("║ [2] abort this consult — worker proceeds without help");
  console.log("║ [3] type a free-form answer yourself");
  console.log("╚═════════════════════════════════════════════════");

  const choice = ((await promptLine("> ")) ?? "").trim();
  if (choice === "2") {
    return harness.resumeConsult(state, {
      kind: "budget_exhausted",
      message: "advisor budget exhausted; proceed without further help",
    });
  }
  if (choice === "3") {
    const text = (await promptLine("answer> ")) ?? "";
    return harness.resumeConsult(state, { kind: "answer", text });
  }
  // Default ([1] or empty): run the advisor handler once more host-side and inject
  // its answer — a bounded escape hatch past the per-kind budget.
  console.log("(running advisor for one more turn…)");
  const childTask = newTask(
    renderConsultInstruction(request),
    SessionId.generate(),
    reactPerLoop(16),
  );
  const r = await entry.handler.run({ task: childTask });
  const answer =
    r.kind === "success"
      ? r.output
      : `advisor did not complete cleanly: ${JSON.stringify(r)}`;
  console.log(`advisor: ${truncate(answer, 300)}`);
  return harness.resumeConsult(state, { kind: "answer", text: answer });
}

/**
 * Present a `budget_exhausted` pause and resume with the operator's choice. The
 * composed tree surfaces a runaway node here under `surface_to_human`; we offer
 * its `available_actions` and resume by re-resolving handles (no
 * reconfiguration).
 */
async function handleHumanEscalation(
  harness: StandardHarness,
  state: PausedState,
  request: HumanRequest,
): Promise<RunResult> {
  if (request.kind !== "budget_exhausted") {
    // The composed tree only escalates via budget_exhausted; anything else is
    // unexpected — halt cleanly.
    console.error(`\nunexpected human request: ${JSON.stringify(request)}`);
    return harness.resume(state, { kind: "halt" });
  }
  const { phase, available_actions: actions } = request;

  console.log(`\n╔═ BUDGET ESCALATION (${phase}) ═══════════════════`);
  actions.forEach((a, i) => console.log(`║ [${i + 1}] ${describeAction(a)}`));
  console.log("╚═════════════════════════════════════════════════");

  const choice = ((await promptLine("> ")) ?? "").trim();
  const idx = Math.max(0, (Number.parseInt(choice, 10) || 1) - 1);
  // Default to a small budget bump so an empty line keeps the run going.
  const action: EscalationAction = actions[idx] ?? {
    kind: "continue_with_budget",
    steps: 12,
  };

  console.log(`(resuming with ${describeAction(action)})`);
  const response: HumanResponse = { kind: "escalate", action };
  return harness.resume(state, response);
}

function describeAction(a: EscalationAction): string {
  switch (a.kind) {
    case "continue_with_budget":
      return `continue with +${a.steps} steps`;
    case "skip":
      return "skip this task";
    case "fail":
      return "fail this node";
  }
}

/** Read one audit prompt from the REPL. `string` to run (empty line ⇒ the default
 *  verbatim); `undefined` on EOF (Ctrl-D), which quits the REPL. */
async function readAuditPrompt(): Promise<string | undefined> {
  console.log(
    "Default audit prompt (press enter to accept, type your own, or Ctrl-D to quit):",
  );
  console.log(`  ${DEFAULT_AUDIT_PROMPT}`);
  const line = await promptLine("audit> ");
  if (line === undefined) return undefined;
  return line.trim() === "" ? DEFAULT_AUDIT_PROMPT : line;
}

/** Print a prompt and read one line from stdin (trailing newline stripped).
 *  `undefined` on EOF. */
async function promptLine(prompt: string): Promise<string | undefined> {
  const rl = createInterface({ input, output });
  try {
    const answer = await rl.question(prompt);
    return answer.replace(/[\r\n]+$/, "");
  } catch {
    return undefined;
  } finally {
    rl.close();
  }
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

/** Run `main` only when this module is the program entrypoint — NOT when it is
 *  imported (e.g. by the composition tests, which reuse `buildRegistry` /
 *  `buildTask` / `execEvaluator`). */
function isEntrypoint(): boolean {
  const arg1 = process.argv[1];
  if (arg1 === undefined) return false;
  return import.meta.url === pathToFileURL(arg1).href;
}

if (isEntrypoint()) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
