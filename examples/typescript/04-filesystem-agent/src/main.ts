/**
 * spore-core example 04 — ReAct with the built-in catalogue file tools.
 *
 * This is [`03-tool-use`](../../03-tool-use) with one substantive change. In 03
 * the agent's tools were hand-rolled: we implemented the harness-loop
 * `ToolRegistry` ourselves and dispatched each call by hand. Here we register
 * spore-core's *built-in* catalogue instead — a single builder line:
 *
 * ```ts
 * .tools(StandardTools.codingSet())   // read_file, write_file, list_dir, …
 * ```
 *
 * Everything else is the same: the same `conversational(model)` builder, the
 * same ReAct loop, the same stream-printed `think · turn N` / tool-call output.
 * The thesis of this example is exactly that: **the harness doesn't change —
 * only the registration path does.**
 *
 * ## What it shows
 *
 * - **Catalogue registration.** `.tools(StandardTools.codingSet())` advertises
 *   and dispatches `read_file` / `write_file` / `list_dir` (and friends) with no
 *   bespoke code.
 * - **A real sandbox.** Catalogue file tools go through a sandbox, so unlike 03's
 *   pure-compute tools (which were happy with the default `NullSandbox`) this
 *   example wires a `WorkspaceScopedSandbox` scoped to `sample-files/`.
 * - **A side effect that outlives the process.** The agent writes `SUMMARY.md`
 *   into `sample-files/`. It is still there after the program exits.
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

import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  HarnessBuilder,
  OLLAMA_DEFAULT_BASE_URL,
  OllamaModelInterface,
  SessionId,
  WorkspaceScopedSandbox,
  newTask,
  reactPerLoop,
  type HarnessStreamEvent,
} from "@spore/core";
import { StandardTools } from "@spore/tools";

const SYSTEM_PROMPT =
  "You are a file-summarizing agent. Use list_dir to find files, " +
  "read_file to read each, and write_file to create SUMMARY.md. " +
  "Act using tools — do not just describe.";

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const modelId =
    argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "llama3.2";
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;

  // The agent operates inside the shipped `sample-files/` directory. Resolve it
  // relative to this source file so `pnpm start` works from anywhere; the
  // sandbox canonicalizes the root at construction.
  const here = dirname(fileURLToPath(import.meta.url));
  const workspaceRoot = join(here, "..", "sample-files");

  const prompt =
    argValue(args, "--prompt") ??
    "There are several .txt files in this directory. Use list_dir to find them, " +
      "read_file to read each one, then write a SUMMARY.md containing a one-sentence " +
      "summary of every file. Use write_file to create it.";

  // Same `conversational` harness as 03 — the ONLY substantive change is that we
  // register the built-in catalogue (`.tools(...)`) over a real sandbox instead
  // of hand-rolling a `ToolRegistry`.
  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const sandbox = new WorkspaceScopedSandbox({ root: workspaceRoot });
  const harness = HarnessBuilder.conversational(model)
    .sandbox(sandbox)
    .tools(StandardTools.codingSet())
    .systemPrompt(SYSTEM_PROMPT)
    .build();

  const task = newTask(prompt, SessionId.generate(), reactPerLoop(8));

  console.log(`model  : ${modelId}`);
  console.log(`dir    : ${workspaceRoot}`);
  console.log(`prompt : ${prompt}\n`);

  // Print each turn (Think) and each catalogue tool call + result (Act /
  // Observe). Because the catalogue dispatches internally, the Act/Observe lines
  // come from harness STREAM events, not from inside a hand-rolled dispatch like
  // 03.
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
    const summary = join(workspaceRoot, "SUMMARY.md");
    if (existsSync(summary)) {
      console.log(`\nSUMMARY.md now exists on disk: ${summary}`);
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

/** Keep observe lines readable — file contents can be long. */
function truncate(s: string, max: number): string {
  const flat = s.replace(/\n/g, " ");
  const chars = [...flat];
  return chars.length <= max ? flat : `${chars.slice(0, max).join("")}…`;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
