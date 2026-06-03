/**
 * spore-core example 06 — web research with an external API tool.
 *
 * This is the first example whose tools reach **outside the process** to a
 * third-party HTTP service. The whole point is that this changes *nothing* about
 * the harness: an external API is just another tool. It drops into the exact same
 * `conversational(model)` builder, the same ReAct loop, and the same
 * `WorkspaceScopedSandbox` you saw in [`04-filesystem-agent`](../../04-filesystem-agent).
 *
 * Tools wired (all from the built-in catalogue, no custom `Tool` impl):
 *
 * - `web_search` — {@link StandardTools.webSearchWithEndpoint}. The query is
 *   POSTed to `endpoint` as JSON `{ "query": ... }` and the response body is
 *   returned to the agent verbatim. The endpoint comes from
 *   `SPORE_WEB_SEARCH_ENDPOINT` (see the README + `.env.example`).
 * - `write_file` — {@link StandardTools.writeFile}. The agent writes its
 *   synthesized, cited answer to `answer.md`.
 * - `read_file` — {@link StandardTools.readFile}. Lets the agent re-read what it
 *   wrote (e.g. to verify or revise the answer).
 *
 * The ONLY substantive difference from 04 is the tool set: 04 registers
 * `codingSet()`, 06 registers `webSearchWithEndpoint(..)` + `writeFile` +
 * `readFile`. Same `conversational` harness, same `WorkspaceScopedSandbox`
 * (here scoped to this example's `workspace/` dir so `write_file` cannot escape
 * it). 04 wrote `SUMMARY.md`; 06 writes `answer.md` — and, like 04, the agent
 * writes it INSIDE the ReAct loop, never `main`.
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
  newTask,
  type HarnessStreamEvent,
} from "@spore/core";
import { StandardTools } from "@spore/tools";

const SYSTEM_PROMPT =
  "You are a web-research agent. Use web_search to find current information, " +
  "synthesize what you learn into a clear answer, and ALWAYS cite the sources " +
  "you used. Write the final answer to answer.md using write_file. Act using " +
  "tools — do not answer from memory alone.";

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
    // A TIMELESS research question: the answer evolves over time but the
    // question stays interesting and is not tied to a single news event.
    "What is the current recommended way to install Rust on macOS, and what are " +
      "the main alternatives? Search the web, synthesize the options, cite your " +
      "sources, and write the answer to answer.md.";

  // Same `conversational` harness + `WorkspaceScopedSandbox` as 04. The ONLY
  // substantive change is the tool set: `web_search` (external API) composes with
  // `write_file` / `read_file` in one builder chain. `.tool()` pushes into the
  // same registry with last-wins upsert by name.
  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const sandbox = new WorkspaceScopedSandbox({ root: workspaceRoot });
  const harness = HarnessBuilder.conversational(model)
    .sandbox(sandbox) // same as 04
    .tool(StandardTools.webSearchWithEndpoint(endpoint)) // ← external API
    .tool(StandardTools.writeFile()) // ← writes answer.md
    .tool(StandardTools.readFile())
    .systemPrompt(SYSTEM_PROMPT)
    .build();

  const task = newTask(prompt, SessionId.generate(), {
    kind: "re_act",
    max_iterations: 10,
  });

  console.log(`model    : ${modelId}`);
  console.log(`endpoint : ${endpoint}`);
  console.log(`workspace: ${workspaceRoot}`);
  console.log(`prompt   : ${prompt}\n`);

  // Print each turn (Think) and each tool call + result (Act / Observe). The
  // search queries and result snippets show up here because `web_search`
  // dispatches through the harness like any other catalogue tool.
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
    const answer = join(workspaceRoot, "answer.md");
    if (existsSync(answer)) {
      console.log(`\nanswer.md now exists on disk: ${answer}`);
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
