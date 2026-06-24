/**
 * spore-core example 08 — multi-step goal decomposition with PlanExecute.
 *
 * This is the first example to swap the **loop strategy**. Everything else —
 * the `conversational(model)` builder, the {@link WorkspaceScopedSandbox}, and
 * the tool set (`web_search` via {@link WebSearchTool.withConfig} + `write_file`
 * + `read_file`, identical to 06) — is
 * held constant. The ONLY substantive change is one line on the `Task`:
 *
 * ```ts
 * // 06 — react step-by-step:
 * newTask(prompt, SessionId.generate(), reactPerLoop(10));
 * // 08 — decompose first via a composed PlanExecute(plan: ReAct, execute: ReAct):
 * newTask(prompt, SessionId.generate(), planExecuteStrategy(), { max_turns: 64 });
 * ```
 *
 * Post-#119, `PlanExecute` is a composed tree, not a flat literal. Its `plan`
 * slot is STRUCTURED — a bare `ReAct` there MUST declare an `output` schema
 * (here `plan-schema`), which the {@link ExecutionRegistry} validates at run
 * entry; the only handle registered explicitly is `plan-schema`.
 *
 * With `plan_execute`, the harness runs one constrained planner turn FIRST: the
 * model must return strict JSON `{ "tasks": [...], "rationale": ... }`. That
 * plan is captured into a {@link PlanArtifact}, surfaced, then each subtask is
 * run in a bounded sub-loop. The turn budget is divided across subtasks
 * (per-task cap = remaining_turns / remaining_tasks), so we set a generous
 * `max_turns`.
 *
 * ## Surfacing the plan — via lifecycle HOOKS, not stream events
 *
 * There are no plan/subtask *stream* events; the plan is visible through the
 * hook chain. We register a {@link PlanExecuteReporter} ({@link Hook}) on two
 * events:
 *
 * - `on_plan_created` fires post-capture / pre-execute — we print a `── plan ──`
 *   banner: the rationale, then the numbered tasks.
 * - `on_task_advance` fires before each subtask — we print `[i/N] <instruction>`
 *   (i = `task_index + 1`); the subtask instruction is `task.instruction`.
 *
 * Tools wired (all from the built-in catalogue, identical to 06):
 *
 * - `web_search` — {@link WebSearchTool.withConfig}; query issued as
 *   `GET <SPORE_WEB_SEARCH_ENDPOINT>?q=<query>` against a SearXNG JSON API.
 * - `write_file` — the agent writes `async-comparison.md` into `workspace/`.
 * - `read_file` — lets the agent re-read what it wrote.
 *
 * ## Staying within small (~128K) context windows
 *
 * Under PlanExecute, verbose tool output is retained across every plan step, so
 * a few searches can overflow a model with a ~128K window (e.g. `gemma4:e4b`,
 * 131072 tokens). Two measures keep this example running cleanly on such models:
 *
 * - It **distills `web_search` results**: a {@link ConciseWebSearch} wrapper
 *   trims the verbatim SearXNG JSON (25-40K tokens/call) down to the top 6
 *   results with only `title` / `url` / `content`, so context stays small.
 * - It **lowers the compaction threshold** to `0.45` (compaction at ≈90K tokens
 *   instead of the default ≈160K), installed via `.contextManager(...)`, so
 *   compaction fires before a 128K-window model overflows.
 *
 * ## Run it
 *
 * ```sh
 * ollama serve &
 * ollama pull llama3.2
 * pnpm install
 * export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"  # SearXNG JSON API
 * pnpm start
 * ```
 */

import { existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  ExecutionRegistry,
  HarnessBuilder,
  OLLAMA_DEFAULT_BASE_URL,
  OllamaModelInterface,
  SessionId,
  WorkspaceScopedSandbox,
  cacheProvider,
  context,
  hooks,
  newTask,
  reactPerLoop,
  toolRegistry,
  type HarnessStreamEvent,
  type LoopStrategy,
  type SandboxProvider,
  type ToolCall,
  type ToolOutput,
} from "@spore/core";
import { StandardTools, WebSearchTool } from "@spore/tools";

type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;

// The GLOBAL operating prompt — the shared capability contract. It is the
// DEFAULT every leaf falls back to. Under SC-10 (#161) the plan and execute
// leaves below each carry their OWN `system_prompt`, which REPLACES this for
// those phases (each phase sees ONLY its own prompt). This global remains the
// prompt any leaf WITHOUT an override would use.
const SYSTEM_PROMPT =
  "You are a research-and-writing agent. Your ONLY capabilities are: web_search " +
  "(find current information online), read_file, and write_file (save your work " +
  "to the workspace). You have NO shell or terminal — you cannot install software, " +
  "set up projects or environments, run/compile/build code, or execute commands. " +
  "Act using tools — do not answer from memory alone.";

// SC-10: the PLAN phase's own system prompt. The planner only DECOMPOSES — it
// never executes a subtask, so its prompt is about producing a good plan, not
// about searching/writing. This replaces SYSTEM_PROMPT for the plan leaf only.
const PLAN_SYSTEM_PROMPT =
  "You are the PLANNER. Your ONLY job is to decompose the goal into an ordered " +
  "list of subtasks. Each subtask must be achievable with web_search and " +
  "write_file alone — there is NO shell or terminal, so never plan setup, " +
  "installation, or build steps. Do not perform any subtask yourself; output " +
  "ONLY the plan.";

// SC-10: the EXECUTE phase's own system prompt. The executor works ONE subtask
// at a time — it does not re-plan. This replaces SYSTEM_PROMPT for the execute
// leaf only, so plan-phase decomposition guidance never leaks into execution.
const EXECUTE_SYSTEM_PROMPT =
  "You are the EXECUTOR. You are given ONE subtask at a time. Use web_search to " +
  "gather current information for it, then synthesize a clear, cited result and " +
  "save your work with write_file. Do not re-plan or invent new subtasks — " +
  "complete the one you were given, using tools rather than memory.";

/**
 * The plan-phase output contract (`plan-schema`). Post-#119, `PlanExecute`'s
 * `plan` slot is STRUCTURED: a bare `ReAct` there must declare an `output`
 * schema so the slot yields a typed task graph ({@link ExecutionRegistry.validate}
 * enforces this via its structured-slot check). This is the strict JSON the
 * planner turn returns.
 */
function planSchema(): unknown {
  return {
    type: "object",
    properties: {
      tasks: {
        type: "array",
        description: "Ordered subtasks to execute in sequence.",
        items: { type: "string" },
      },
      rationale: { type: "string" },
    },
    required: ["tasks"],
  };
}

/**
 * The {@link ExecutionRegistry} the composed strategy's handles resolve against.
 * The single `plan-schema` slot is the only EXPLICIT handle; the builder
 * default-fills the empty agent/toolset handles (`reactPerLoop` uses empty
 * handles) from the harness's own model and global tool catalogue at `build`.
 */
export function buildRegistry(): ExecutionRegistry {
  return ExecutionRegistry.builder().schema("plan-schema", planSchema()).build();
}

/**
 * The post-#119 composed strategy: `PlanExecute(plan: ReAct, execute: ReAct)`.
 * The plan leaf carries the `plan-schema` output contract (required for the
 * structured `plan` slot); both leaves use empty agent/toolset handles that the
 * builder default-fills. The plan/execute leaves stay effectively unbounded
 * (`reactPerLoop(Number.MAX_SAFE_INTEGER)`, the TS analogue of Rust's
 * `u32::MAX`) so the global `max_turns` backstop governs. Old flat shape was
 * `{ kind: "plan_execute" }`.
 *
 * SC-10 (#161, per-leaf system prompt): the plan and execute leaves each carry
 * their OWN `system_prompt`. The plan phase runs under `PLAN_SYSTEM_PROMPT`
 * (decompose only) and the execute phase under `EXECUTE_SYSTEM_PROMPT` (do one
 * subtask) — each phase sees ONLY its own prompt, so planning guidance never
 * leaks into execution and vice versa. (The per-leaf TOOLSET override is the
 * existing `ReactConfig.toolset` handle; here both phases share the global
 * catalogue.) The global `SYSTEM_PROMPT` remains the documented fallback any
 * leaf WITHOUT an override would use.
 */
export function planExecuteStrategy(): LoopStrategy {
  return {
    kind: "plan_execute",
    plan: {
      ...reactPerLoop(Number.MAX_SAFE_INTEGER),
      output: "plan-schema",
      system_prompt: PLAN_SYSTEM_PROMPT,
    },
    execute: {
      ...reactPerLoop(Number.MAX_SAFE_INTEGER),
      system_prompt: EXECUTE_SYSTEM_PROMPT,
    },
  };
}

/**
 * Lifecycle hook that prints the PlanExecute plan and each subtask as it runs.
 *
 * `on_plan_created` fires once, after the planner turn captures the plan and
 * before any subtask executes — the money moment for PlanExecute.
 * `on_task_advance` fires before each subtask. Both are sync, plan/task-carrying
 * events. This hook only observes; it always returns `{ decision: "continue" }`.
 *
 * It also records the captured subtask count so the run can confirm the plan on
 * success. In the TypeScript port the plan is persisted to the RunStore seam
 * (core #76), not `session_state.extras`, so the hook is the portable view.
 */
class PlanExecuteReporter implements hooks.Hook {
  /** Number of subtasks the planner produced, captured on `on_plan_created`. */
  subtaskCount: number | undefined;

  async handle(ctx: hooks.HookContext): Promise<hooks.HookDecision> {
    if (ctx.event === "on_plan_created") {
      const { plan } = ctx;
      this.subtaskCount = plan.tasks.length;
      console.log("\n── plan ──");
      if (plan.rationale.trim() !== "") {
        console.log(`rationale: ${plan.rationale}`);
      }
      plan.tasks.forEach((task, i) => {
        console.log(`  ${i + 1}. ${task}`);
      });
      console.log("──────────\n");
    } else if (ctx.event === "on_task_advance") {
      console.log(
        `[${ctx.task_index + 1}/${ctx.total_tasks}] ${ctx.task.instruction}`,
      );
    }
    return { decision: "continue" };
  }

  events(): hooks.HookEvent[] {
    return ["on_plan_created", "on_task_advance"];
  }

  name(): string {
    return "plan-execute-reporter";
  }
}

/**
 * A thin wrapper around the built-in {@link WebSearchTool} that distills its
 * output.
 *
 * WHY: the core `web_search` tool returns the SearXNG JSON body VERBATIM (by
 * frozen spec — normalization is out of scope for the core tool). Each search
 * yields ~25-30 results, each carrying the full `content` plus a dozen noise
 * fields (`thumbnail`, `engine`, `score`, `parsed_url`, …) — roughly 25-40K
 * tokens per call. Under PlanExecute those dumps are retained across every plan
 * step, so three searches alone can overflow a ~128K-window model. This wrapper
 * keeps only the top results and the fields the agent actually reads, so the
 * conversation context stays small. The model still sees an identical
 * `web_search` tool (same name + schema); only the *result* is trimmed.
 */
class ConciseWebSearch implements Tool {
  readonly name = "web_search";

  constructor(private readonly inner: WebSearchTool) {}

  mayProduceLargeOutput(): boolean {
    return true;
  }

  async execute(
    call: ToolCall,
    sandbox: SandboxProvider,
    ctx: ToolContext,
    signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const out = await this.inner.execute(call, sandbox, ctx, signal);
    // Errors and every non-success variant pass through untouched.
    if (out.kind !== "success") {
      return out;
    }
    return {
      kind: "success",
      content: distillSearchResults(out.content),
      truncated: out.truncated,
    };
  }
}

/**
 * Keep only the top 6 results, and for each only `title` / `url` / `content`
 * (content clipped to ~500 chars). Drop all other fields and top-level keys
 * (`answers`, `infoboxes`, `suggestions`, `unresponsive_engines`, …), and
 * re-serialize as compact `{"results":[...]}`. Defensive: if the body is not
 * JSON or has no `results` array, the original string is returned unchanged —
 * we never error just because the shape was unexpected.
 */
function distillSearchResults(content: string): string {
  let value: unknown;
  try {
    value = JSON.parse(content);
  } catch {
    return content;
  }
  const results =
    typeof value === "object" && value !== null
      ? (value as Record<string, unknown>).results
      : undefined;
  if (!Array.isArray(results)) {
    return content;
  }

  const distilled = results.slice(0, 6).map((r) => {
    const rec = (typeof r === "object" && r !== null ? r : {}) as Record<
      string,
      unknown
    >;
    const title = typeof rec.title === "string" ? rec.title : "";
    const url = typeof rec.url === "string" ? rec.url : "";
    const body = typeof rec.content === "string" ? rec.content : "";
    return { title, url, content: [...body].slice(0, 500).join("") };
  });

  return JSON.stringify({ results: distilled });
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const modelId =
    argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "llama3.2";
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;

  // Opt-in constrained decoding. OFF by default: tool-capable models (incl.
  // `*-cloud` like `gemma4:31b-cloud`) use native Ollama tool calling, which
  // gives `write_file` a real typed schema and no always-on `final` escape.
  // Small local models (e.g. `llama3.2`) that leak `<|python_tag|>` or malformed
  // JSON can pass `--structured` to force the JSON-object channel.
  const structured = args.includes("--structured");

  // The search backend endpoint. `web_search` issues `GET <endpoint>?q=<query>`
  // and returns the JSON body to the agent. There is no live backend in
  // spore-core, so you must supply one — a self-hosted SearXNG JSON API. The
  // endpoint already carries `format=json`; the GET path preserves it and
  // appends the `q` param (core #108, now implemented). Brave/Tavily-style auth
  // headers are also supported via `WebSearchTool.withConfig` (`authHeaders` /
  // `bodyAuthParams`). See README.
  const endpoint = process.env.SPORE_WEB_SEARCH_ENDPOINT?.trim();
  if (!endpoint) {
    console.error(
      "SPORE_WEB_SEARCH_ENDPOINT is not set.\n" +
        "Set it to a SearXNG JSON endpoint, e.g.\n" +
        '  export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"\n' +
        "See .env.example and the README.",
    );
    process.exit(2);
  }

  // The agent operates inside this example's `workspace/` directory. Resolve it
  // relative to this source file so `pnpm start` works from anywhere, and create
  // it if missing — the sandbox requires an existing root, which it canonicalizes
  // at construction.
  const here = dirname(fileURLToPath(import.meta.url));
  const workspaceRoot = join(here, "..", "workspace");
  mkdirSync(workspaceRoot, { recursive: true });

  const prompt =
    argValue(args, "--prompt") ??
    // A multi-step goal that benefits from upfront decomposition: search each
    // runtime, synthesize a comparison, then write the file.
    "Research the Rust async ecosystem, write a comparison of tokio vs " +
      "async-std vs smol covering performance, ecosystem maturity, and use " +
      "cases, and save it to async-comparison.md.";

  // Register the plan reporter on a StandardHookChain. The chain is how the plan
  // becomes visible: there are no plan/subtask stream events.
  const chain = new hooks.StandardHookChain();
  const reporter = new PlanExecuteReporter();
  chain.register(reporter);

  // Same `conversational` harness + `WorkspaceScopedSandbox` + tool set as 06.
  // The ONLY substantive change vs 06 is the loop strategy on the Task below.
  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const sandbox = new WorkspaceScopedSandbox({ root: workspaceRoot });

  // Lower the compaction threshold so it fires at ~0.45 * 200K ≈ 90K tokens,
  // BEFORE a ~128K-window model (e.g. `gemma4:e4b`) overflows. `conversational`
  // installs a StandardContextManager with default compaction (compaction at 80%
  // of a 200K window ≈ 160K), which is too late here. The context manager gets
  // its OWN raw model instance for summarization turns (it advertises no tools).
  const contextManager = context.intoHarnessAdapter(
    new context.StandardContextManager(
      OllamaModelInterface.withBaseUrl(modelId, baseUrl),
      new cacheProvider.NullCacheProvider(),
      { ...context.defaultCompactionConfig(), threshold: 0.45 },
    ),
  );

  const harness = HarnessBuilder.conversational(model)
    .sandbox(sandbox) // same as 06
    .contextManager(contextManager)
    // The composed strategy's handles resolve against this registry at run entry
    // (`validate()`). Only `plan-schema` is explicit; the empty agent/toolset
    // handles of the bare `reactPerLoop` leaves are default-filled from this
    // harness's own model + tool catalogue at `build`.
    .registry(buildRegistry())
    .tool({
      // GET <endpoint>?q=<query>; the endpoint's `?format=json` is preserved.
      // Wrapped in ConciseWebSearch so verbose SearXNG JSON is distilled before
      // it enters the conversation (see the struct doc above). Same name +
      // schema, so the model sees an identical `web_search` tool.
      implementation: new ConciseWebSearch(
        WebSearchTool.withConfig({
          endpoint,
          method: "GET",
          queryParam: "q",
          authHeaders: [],
          bodyAuthParams: [],
        }),
      ),
      schema: WebSearchTool.schema(),
    })
    .tool(StandardTools.writeFile())
    .tool(StandardTools.readFile())
    .systemPrompt(SYSTEM_PROMPT)
    // Native tool calling by default; `--structured` flips on constrained
    // decoding for small models (see the `structured` flag above). With
    // structured mode the "think · turn N" line is just a turn marker, not model
    // chatter, since each turn emits one clean JSON tool call across both the
    // plan and execute phases.
    .modelParams({ structured_tool_calls: structured, stop_sequences: [] })
    .hooks(chain) // ← the plan becomes visible through the hook chain
    .build();

  // THE STRATEGY SWAP. 06 used a bare `reactPerLoop(10)`; here we decompose first
  // via the composed `PlanExecute(plan: ReAct{plan-schema}, execute: ReAct)` tree
  // (post-#119 recursive union). The plan leaf carries the `plan-schema` output
  // contract — `PlanExecute`'s `plan` slot is STRUCTURED, so a bare `ReAct` there
  // MUST declare an output schema (`registry.validate()` enforces this). The turn
  // budget is divided across subtasks, so we give it generous headroom via
  // `max_turns`.
  const task = newTask(prompt, SessionId.generate(), planExecuteStrategy(), {
    max_turns: 64,
  });

  console.log(`model    : ${modelId}`);
  console.log(`endpoint : ${endpoint}`);
  console.log(`workspace: ${workspaceRoot}`);
  console.log(`strategy : PlanExecute (06 used ReAct)`);
  console.log(`prompt   : ${prompt}\n`);

  // Print each turn (Think) and each tool call + result (Act / Observe). Unlike
  // the Rust/Python/Go ports, the TypeScript harness streams the subtask inner
  // tool calls too — so these lines show both the plan-phase turn and each
  // subtask's tool activity. The hooks above remain the portable, cross-language
  // view of the plan and subtask boundaries.
  const onStream = (event: HarnessStreamEvent): void => {
    switch (event.kind) {
      case "turn_start":
        console.log(`think  · turn ${event.turn}`);
        break;
      case "tool_call":
        console.log(
          `    act    → ${event.name}(${JSON.stringify(event.args ?? {})})`,
        );
        break;
      case "tool_result": {
        const tag = event.is_error ? "obs(err)" : "obs ";
        console.log(`    ${tag}→ ${truncate(event.content ?? "", 200)}`);
        break;
      }
      default:
        break;
    }
  };

  let result;
  try {
    result = await harness.run({ task, on_stream: onStream });
  } catch (err) {
    console.error(
      `\ncould not reach the model — is Ollama running at ${baseUrl}? (\`ollama serve\`)\n${err instanceof Error ? err.message : String(err)}`,
    );
    process.exit(1);
  }

  if (result.kind === "success") {
    console.log(`\nanswer (${result.turns} turn(s)): ${result.output}`);
    // The captured plan was surfaced through the hook chain; confirm its size.
    if (reporter.subtaskCount !== undefined) {
      console.log(`\nplan had ${reporter.subtaskCount} subtask(s)`);
    }
    const doc = join(workspaceRoot, "async-comparison.md");
    if (existsSync(doc)) {
      console.log(`\nasync-comparison.md now exists on disk: ${doc}`);
    }
    return;
  }
  console.error(`\nrun did not succeed: ${JSON.stringify(result)}`);
  process.exit(1);
}

function argValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

/** Keep observe lines readable — search results can be long. */
function truncate(s: string, max: number): string {
  const flat = s.replace(/\n/g, " ");
  const chars = [...flat];
  return chars.length <= max ? flat : `${chars.slice(0, max).join("")}…`;
}

/** Run `main` only when this module is the program entrypoint — NOT when it is
 *  imported (e.g. by the composition test, which reuses `buildRegistry` /
 *  `planExecuteStrategy`). */
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
