/**
 * spore-core example 08 — multi-step goal decomposition with PlanExecute.
 *
 * This is the first example to swap the **loop strategy**. Everything else —
 * the `conversational(model)` builder, the {@link WorkspaceScopedSandbox}, and
 * the tool set (`web_search` + `write_file` + `read_file`, identical to 06) — is
 * held constant. The ONLY substantive change is one line on the `Task`:
 *
 * ```ts
 * // 06 — react step-by-step:
 * newTask(prompt, SessionId.generate(), { kind: "re_act", max_iterations: 10 });
 * // 08 — decompose the goal first, then execute each subtask:
 * newTask(prompt, SessionId.generate(), { kind: "plan_execute" }, { max_turns: 24 });
 * ```
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
 * - `web_search` — {@link StandardTools.webSearchWithEndpoint}; query POSTed to
 *   `SPORE_WEB_SEARCH_ENDPOINT` as JSON `{ "query": ... }`.
 * - `write_file` — the agent writes `async-comparison.md` into `workspace/`.
 * - `read_file` — lets the agent re-read what it wrote.
 *
 * ## Run it
 *
 * ```sh
 * ollama serve &
 * ollama pull llama3.2
 * pnpm install
 * export SPORE_WEB_SEARCH_ENDPOINT=http://localhost:8888/search  # a {"query"}->JSON endpoint
 * pnpm start
 * ```
 */

import { existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  HarnessBuilder,
  OLLAMA_DEFAULT_BASE_URL,
  OllamaModelInterface,
  SessionId,
  WorkspaceScopedSandbox,
  hooks,
  newTask,
  type HarnessStreamEvent,
} from "@spore/core";
import { StandardTools } from "@spore/tools";

const SYSTEM_PROMPT =
  "You are a planning research agent. Decompose the goal into clear subtasks. " +
  "For each subtask, use web_search to find current information, then " +
  "synthesize a clear, cited comparison and save the final document with " +
  "write_file. Act using tools — do not answer from memory alone.";

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

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const modelId =
    argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "llama3.2";
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;

  // The search backend endpoint. `web_search` POSTs `{ "query": ... }` here and
  // returns the JSON body to the agent. There is no live backend in spore-core,
  // so you must supply one — a self-hosted SearXNG JSON endpoint, or a mock that
  // accepts the `{ "query" }` shape. Raw Brave/Tavily are NOT yet drop-in: they
  // need a custom auth header, which is tracked as core issue #108. See README.
  const endpoint = process.env.SPORE_WEB_SEARCH_ENDPOINT?.trim();
  if (!endpoint) {
    console.error(
      "SPORE_WEB_SEARCH_ENDPOINT is not set.\n" +
        'Set it to a search endpoint that accepts a JSON `{"query": ...}` POST ' +
        "and returns JSON results.\n" +
        "See .env.example and the README. (Raw Brave/Tavily need core #108 first.)",
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
  const harness = HarnessBuilder.conversational(model)
    .sandbox(sandbox) // same as 06
    .tool(StandardTools.webSearchWithEndpoint(endpoint))
    .tool(StandardTools.writeFile())
    .tool(StandardTools.readFile())
    .systemPrompt(SYSTEM_PROMPT)
    .hooks(chain) // ← the plan becomes visible through the hook chain
    .build();

  // THE ONE-LINE SWAP. 06 used `{ kind: "re_act", max_iterations: 10 }`; here we
  // decompose first via `{ kind: "plan_execute" }`. The turn budget is divided
  // across subtasks, so we give it generous headroom via `max_turns`.
  const task = newTask(
    prompt,
    SessionId.generate(),
    { kind: "plan_execute" },
    { max_turns: 24 },
  );

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

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
