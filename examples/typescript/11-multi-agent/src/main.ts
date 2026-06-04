/**
 * spore-core example 11 — multi-agent composition.
 *
 * **The thesis: agents are composable.** The harness does not care whether a
 * "tool" dispatches to a function or to *another agent*. This example builds
 * three agents and wires two of them into the third as ordinary tools.
 *
 * ## The three agents
 *
 * - **research worker** — a {@link HarnessBuilder.conversational} harness with
 *   exactly one tool, `web_search` (the SearXNG-backed {@link WebSearchTool}
 *   from example 06). Given an instruction string, it searches the web and
 *   returns raw, cited findings as its output string.
 * - **writing worker** — a harness with NO tools. Given the research findings as
 *   its instruction, it formats them into a polished markdown report and returns
 *   that markdown as its output string. It never touches the network — it only
 *   shapes prose.
 * - **orchestrator** — a harness whose "tools" are the two workers (wrapped as
 *   {@link SubagentTool}s) plus `write_file`. It plans the job, calls
 *   `research_worker`, hands that output to `writing_worker`, then writes the
 *   final markdown to `workspace/report.md`.
 *
 * Three agents, two handoffs (`research → writing`, `writing → report.md`),
 * one output.
 *
 * ## The agent-as-tool mechanism
 *
 * Each worker is a fully-built child {@link Harness} wrapped in a
 * {@link SubagentTool}. `SubagentTool` implements the same `Tool` interface as
 * `write_file` or `web_search`: when the orchestrator emits a `tool_call` for
 * `research_worker`, the tool reads a single `instruction` string from the call,
 * runs the child harness on a fresh `Task`, and returns the child's final output
 * string as the tool result. The orchestrator cannot tell — and does not need to
 * know — that the "tool" behind `research_worker` is an entire agent with its own
 * loop, its own model, and its own web-search tool.
 *
 * We register each worker on the orchestrator's builder the same way example 06
 * registers `web_search`: build a {@link StandardTool} (`{ implementation,
 * schema }`) from the `SubagentTool` plus a schema advertising the
 * `{ instruction: string }` input, then `.tool(...)` it.
 *
 * ## Why this keeps the orchestrator's context clean
 *
 * Both workers use isolated context sharing (`{ kind: "isolated" }`): each runs
 * in a brand-new session with NO shared mutable state with the orchestrator or
 * with each other. The research worker may burn a dozen internal turns issuing
 * search queries and sifting noisy JSON — but the ONLY thing that crosses back
 * into the orchestrator's context is the worker's final output string. The
 * orchestrator never sees the worker's intermediate turns, failed searches, or
 * raw result blobs. The noisy work is encapsulated; the orchestrator's context
 * stays small and on-topic. This is the whole reason to delegate to a subagent
 * rather than inline the work.
 *
 * A direct, visible consequence: the child's internal turns do **not** stream up
 * through the parent. The orchestrator's stream only shows the `tool_call` to
 * `research_worker` and the `tool_result` coming back — which is exactly the
 * agent boundary we print. The invisibility of the child's turns is not a
 * limitation; it *is* the context isolation, made observable.
 *
 * ## The strategy split: PlanExecute at the top, ReAct inside
 *
 * The orchestrator runs `{ kind: "plan_execute" }`: it decomposes the job
 * ("research, then write, then save") into subtasks up front and executes them
 * in order — natural for a coordinator. Each worker, by contrast, runs ReAct
 * internally. (The ReAct loop is hardcoded inside `SubagentTool`; a subagent
 * always runs its child as `re_act`.) So the two layers use two different loop
 * strategies, each fit to its level: deliberate planning at the orchestrator,
 * step-by-step tool use inside the workers.
 *
 * ## Agent boundaries in stdout
 *
 * The point of this example is *legibility*: you should be able to read stdout
 * and see which agent is acting, what it received, and what it returned. The
 * orchestrator's stream fires a `tool_call` and a `tool_result` for each worker
 * dispatch — we turn those into a boxed banner:
 *
 * ```text
 * ┌─ orchestrator → research_worker
 * │  received: <instruction>
 * └─ research_worker → orchestrator
 *    returned: <truncated findings>
 * ```
 *
 * The `tool_result` event carries only a `call_id` (no tool name), so we remember
 * which `call_id` belonged to which tool when the `tool_call` fires, then look it
 * up on the result to label the closing half of the boundary.
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
import { fileURLToPath } from "node:url";

import {
  HarnessBuilder,
  OLLAMA_DEFAULT_BASE_URL,
  OllamaModelInterface,
  SessionId,
  WorkspaceScopedSandbox,
  newTask,
  toolRegistry,
  type Harness,
  type HarnessStreamEvent,
} from "@spore/core";
import {
  StandardTools,
  SubagentTool,
  WebSearchTool,
  type ContextSharing,
} from "@spore/tools";

/** Per-worker wall-clock cap. A worker can burn many internal ReAct turns; this
 * bounds how long the orchestrator will wait on any single delegation. */
const WORKER_TIMEOUT_MS = 180_000;

const RESEARCH_PROMPT =
  "You are a research worker. Use the web_search tool to gather current, " +
  "factual information on the topic you are given. Issue focused queries, read " +
  "the results, and return a concise set of findings as plain text — key facts, " +
  "figures, and definitions — each followed by the source URL it came from. Do " +
  "NOT format a report; just return the raw, cited findings. Act using " +
  "web_search — do not answer from memory alone.";

const WRITING_PROMPT =
  "You are a writing worker. You will be given a set of raw, cited research " +
  "findings. Turn them into a polished markdown report: a top-level `# ` title, " +
  "a short intro, well-organized `## ` sections, and a `## Sources` list " +
  "preserving the URLs from the findings. Return ONLY the markdown of the " +
  "report — no preamble, no commentary. You have no tools; produce the report " +
  "directly as your final answer.";

const ORCHESTRATOR_PROMPT =
  "You are an orchestrator. You coordinate two worker agents, each exposed to " +
  "you as a tool. Your plan is always the same three steps: (1) call " +
  "`research_worker` with an `instruction` describing the topic to research; " +
  "(2) call `writing_worker` with an `instruction` that is the EXACT findings " +
  "text returned by the research worker, asking it to format a polished " +
  "markdown report; (3) call `write_file` to save the writing worker's markdown " +
  "verbatim to `report.md`. Do the research and writing by delegating to the " +
  "workers — never do it yourself — and always finish by writing report.md.";

/** The single-parameter input schema every worker tool advertises: the
 * orchestrator passes one `instruction` string, which `SubagentTool` forwards to
 * the child harness as its task. */
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

/** Build the SearXNG-backed `web_search` catalogue tool (identical wiring to
 * example 06). Only the research worker gets this. */
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

/** Build the research worker: a child harness whose only tool is `web_search`.
 * Each agent gets its OWN fresh model instance — the workers are genuinely
 * independent and do not share a model object with the orchestrator. */
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

/** Build the writing worker: a child harness with NO tools — it formats prose
 * and returns the report as its final answer. */
function buildWritingHarness(modelId: string, baseUrl: string): Harness {
  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  return HarnessBuilder.conversational(model)
    .systemPrompt(WRITING_PROMPT)
    .build();
}

/** Wrap a child harness as a `SubagentTool` and bundle it into a `StandardTool`
 * the orchestrator can register — exactly how example 06 wraps `web_search`,
 * only the "implementation" here is an entire agent. */
function buildWorkerTool(
  name: string,
  description: string,
  child: Harness,
): toolRegistry.StandardTool {
  // `childRegistry` is used ONLY for the depth-1 `hasSubagentTools()` check. The
  // workers have no subagent tools of their own, so a fresh empty registry passes
  // trivially. The child's REAL tools were wired on its builder above.
  const emptyChildRegistry = new toolRegistry.StandardToolRegistry();
  const isolated: ContextSharing = { kind: "isolated" };
  const subagent = SubagentTool.buildOrThrow({
    name,
    description,
    inputSchema: instructionSchema(),
    timeoutMs: WORKER_TIMEOUT_MS,
    contextSharing: isolated,
    harness: child,
    childRegistry: emptyChildRegistry,
  });

  return {
    implementation: subagent,
    schema: {
      name,
      description,
      parameters: instructionSchema(),
      // `open_world`: a subagent reaches outside the process (it runs a whole
      // agent, and the research worker hits the network), so it is not a closed,
      // read-only computation.
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
    argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "llama3.2";
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;

  // Same required search backend as example 06 — only the research worker uses
  // it, but the orchestrator cannot do its job without it, so we fail fast here.
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

  // The orchestrator operates inside this example's `workspace/` directory; it
  // writes the final report there. Resolve it relative to this source file so
  // `pnpm start` works from anywhere, and create it if missing — the sandbox
  // requires an existing root, which it canonicalizes at construction.
  const here = dirname(fileURLToPath(import.meta.url));
  const workspaceRoot = join(here, "..", "workspace");
  mkdirSync(workspaceRoot, { recursive: true });

  const topic =
    argValue(args, "--topic") ??
    // A TIMELESS, encyclopedic subject so web-search results stay stable and
    // useful across runs (per the issue: keep the topic generic).
    "the history and core ideas of the Rust programming language";
  const prompt =
    `Research ${topic} and produce a polished markdown report saved to ` +
    "report.md. Delegate the research to research_worker and the writing to " +
    "writing_worker.";

  // ---- Build the two workers, then wrap them as orchestrator tools ----------
  const researchChild = buildResearchHarness(modelId, baseUrl, endpoint);
  const writingChild = buildWritingHarness(modelId, baseUrl);

  const researchTool = buildWorkerTool(
    "research_worker",
    "Delegate to the research agent: pass an `instruction` describing a topic; " +
      "it web-searches and returns concise, cited findings as text.",
    researchChild,
  );
  const writingTool = buildWorkerTool(
    "writing_worker",
    "Delegate to the writing agent: pass an `instruction` containing research " +
      "findings; it returns a polished markdown report.",
    writingChild,
  );

  // ---- Build the orchestrator: workers-as-tools + write_file ----------------
  const orchestratorModel = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const sandbox = new WorkspaceScopedSandbox({ root: workspaceRoot });
  const orchestrator = HarnessBuilder.conversational(orchestratorModel)
    .sandbox(sandbox)
    .tool(researchTool)
    .tool(writingTool)
    .tool(StandardTools.writeFile())
    .systemPrompt(ORCHESTRATOR_PROMPT)
    .build();

  // The orchestrator plans the three steps up front via plan_execute, then
  // executes them. The turn budget is divided across subtasks, so give it
  // generous headroom — each worker dispatch may itself be slow.
  const task = newTask(
    prompt,
    SessionId.generate(),
    { kind: "plan_execute" },
    { max_turns: 32 },
  );

  // The orchestrator's stream is where the agent boundaries become visible. A
  // `tool_call` to a worker IS the "→ worker" boundary; the matching
  // `tool_result` IS the "← worker" boundary. The child's own internal turns do
  // NOT appear here — that invisibility is the context isolation, made
  // observable (see the module docs).
  //
  // `tool_result` carries only `call_id` (no tool name), so we remember which
  // `call_id` belonged to which tool when the `tool_call` fires, then look it up
  // on the result to label the closing half of the boundary.
  const callNames = new Map<string, string>();
  const onStream = (event: HarnessStreamEvent): void => {
    switch (event.kind) {
      case "turn_start":
        console.log(`orchestrator · plan/execute turn ${event.turn}`);
        break;
      case "tool_call": {
        callNames.set(event.call_id, event.name);
        if (isWorker(event.name)) {
          const instruction = readInstruction(event.args);
          console.log(`┌─ orchestrator → ${event.name}`);
          console.log(`│  received: ${truncate(instruction, 200)}`);
        } else {
          console.log(
            `  orchestrator → ${event.name}(${truncate(JSON.stringify(event.args ?? {}), 160)})`,
          );
        }
        break;
      }
      case "tool_result": {
        const name = callNames.get(event.call_id) ?? "<tool>";
        callNames.delete(event.call_id);
        const content = event.content ?? "";
        if (isWorker(name)) {
          const tag = event.is_error ? "FAILED" : "returned";
          console.log(`└─ ${name} → orchestrator`);
          console.log(`   ${tag}: ${truncate(content, 280)}`);
        } else {
          const tag = event.is_error ? "err" : "ok";
          console.log(
            `  ${name} → orchestrator [${tag}]: ${truncate(content, 160)}`,
          );
        }
        break;
      }
      default:
        break;
    }
  };

  console.log(`model     : ${modelId}`);
  console.log(`endpoint  : ${endpoint}`);
  console.log(`workspace : ${workspaceRoot}`);
  console.log("strategy  : orchestrator=PlanExecute, workers=ReAct (isolated)");
  console.log("agents    : orchestrator → [research_worker, writing_worker]");
  console.log(`topic     : ${topic}\n`);

  let result;
  try {
    result = await orchestrator.run({ task, on_stream: onStream });
  } catch (err) {
    console.error(
      `\ncould not reach the model — is Ollama running at ${baseUrl}? (\`ollama serve\`)\n${err instanceof Error ? err.message : String(err)}`,
    );
    process.exit(1);
  }

  if (result.kind === "success") {
    console.log(
      `\norchestrator done (${result.turns} turn(s)): ${truncate(result.output, 280)}`,
    );
    const report = join(workspaceRoot, "report.md");
    if (existsSync(report)) {
      console.log(`\nreport.md now exists on disk: ${report}`);
    } else {
      console.error(
        "\nwarning: orchestrator finished but report.md was not written.",
      );
    }
    return;
  }
  console.error(`\nrun did not succeed: ${JSON.stringify(result)}`);
  process.exit(1);
}

/** A tool name that maps to one of the two worker agents (vs. `write_file`). */
function isWorker(name: string): boolean {
  return name === "research_worker" || name === "writing_worker";
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

/** Keep boundary lines readable — findings and reports can be long. */
function truncate(s: string, max: number): string {
  const flat = s.replace(/\n/g, " ");
  const chars = [...flat];
  return chars.length <= max ? flat : `${chars.slice(0, max).join("")}…`;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
