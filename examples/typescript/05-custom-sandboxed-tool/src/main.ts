/**
 * spore-core example 05 — a custom tool you write yourself.
 *
 * Examples [`03`](../../03-tool-use) and [`04`](../../04-filesystem-agent)
 * showed the two *built-in* tool paths: hand-rolling the harness-loop
 * `ToolRegistry` (03) and registering the shipped catalogue with
 * `.tools(StandardTools.codingSet())` (04). This example shows the third and
 * most important path — **bringing your own tool** — and the public extension
 * point that makes it possible: the `Tool` interface.
 *
 * ## The two custom tools
 *
 * Both are plain classes implementing `Tool` (see `src/tools/`):
 *
 * - **`remember(key, value)`** — persists a fact into the run store. It MUTATES
 *   shared state, so it is not `read_only`.
 * - **`recall(key)`** — reads a fact back out. It only reads, so it is
 *   `read_only` + `idempotent`.
 *
 * ## The seam each tool uses
 *
 * Each tool's `execute(call, sandbox, ctx)` receives two seams: a
 * `SandboxProvider` (the environment — unused here, these tools never touch the
 * filesystem) and a `ToolContext` (the storage seam: `ctx.runStore` +
 * `ctx.sessionId`). Facts are keyed under `fact:{key}` so they cannot collide
 * with reserved catalogue keys.
 *
 * ## The pattern: implement `Tool` → wrap in a `StandardTool` → `.tool()`
 *
 * 1. Implement `Tool` (a class with `name` + `execute`).
 * 2. Bundle the impl with its schema as a `StandardTool` (`{ implementation,
 *    schema }`) so the two can never drift.
 * 3. Register each with `.tool(...)`. The harness wires the sandbox and a
 *    per-run `ToolContext` automatically — **the harness doesn't change, only
 *    what you register does.**
 *
 * Two builder differences from 04: there is no `.tools(...)` catalogue, and no
 * explicit `.sandbox(...)` / `.storage(...)`. `build()` defaults storage to an
 * in-memory provider whenever `.tool()` tools are present, so the run store
 * works for free.
 *
 * ## Run it
 *
 * ```sh
 * ollama serve &
 * ollama pull llama3.2
 * pnpm install
 * pnpm start
 * ```
 */

import {
  HarnessBuilder,
  OLLAMA_DEFAULT_BASE_URL,
  OllamaModelInterface,
  SessionId,
  newTask,
  type HarnessStreamEvent,
  type toolRegistry,
} from "@spore/core";

import { RecallTool } from "./tools/recall.js";
import { RememberTool } from "./tools/remember.js";

type StandardTool = toolRegistry.StandardTool;

const SYSTEM_PROMPT =
  "You are a research agent with a memory. Research the topic the user gives " +
  "you across several turns. As you discover each fact, call `remember` to " +
  "store it under a short, stable key (e.g. 'habitat', 'diet'). Keep track of " +
  "the keys you use. When you have gathered enough facts, call `recall` on each " +
  "key you remembered, then write a final summary built ONLY from the recalled " +
  "facts. Act using tools — do not just describe.";

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const modelId =
    argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "llama3.2";
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;

  const prompt =
    argValue(args, "--prompt") ??
    "Research the common octopus. Remember a few key facts (habitat, diet, " +
      "lifespan, intelligence), then recall them and write a short summary.";

  // Same `conversational` harness as 03 / 04 — the substantive change is that we
  // register two tools WE wrote (`.tool(...)`) instead of a catalogue preset. No
  // `.sandbox(...)` (these tools ignore it) and no `.storage(...)` (build()
  // defaults to in-memory storage when `.tool()` tools are present).
  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const remember: StandardTool = {
    implementation: new RememberTool(),
    schema: RememberTool.schema(),
  };
  const recall: StandardTool = {
    implementation: new RecallTool(),
    schema: RecallTool.schema(),
  };
  const harness = HarnessBuilder.conversational(model)
    .tool(remember)
    .tool(recall)
    .systemPrompt(SYSTEM_PROMPT)
    .build();

  const task = newTask(prompt, SessionId.generate(), {
    kind: "re_act",
    max_iterations: 12,
  });

  console.log(`model  : ${modelId}`);
  console.log(`tools  : remember, recall`);
  console.log(`prompt : ${prompt}\n`);

  // Print each turn (Think) and each tool call + result (Act / Observe) from
  // harness STREAM events — the builder dispatches our tools internally, just as
  // it does the catalogue in 04.
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
    console.log(`\nsummary (${result.turns} turn(s)): ${result.output}`);
    return;
  }
  console.error(`\nrun did not succeed: ${JSON.stringify(result)}`);
  process.exit(1);
}

function argValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

/** Keep observe lines readable — recalled facts can be long. */
function truncate(s: string, max: number): string {
  const flat = s.replace(/\n/g, " ");
  const chars = [...flat];
  return chars.length <= max ? flat : `${chars.slice(0, max).join("")}…`;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
