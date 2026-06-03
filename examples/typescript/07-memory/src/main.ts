/**
 * spore-core example 07 — the storage seam, via a `MarkdownMemoryProvider`.
 *
 * ## What it demonstrates
 * The harness is **stateless**: all durable state lives behind the storage
 * seam. Memory is just one domain of that seam — a `MemoryStore` you implement.
 * The simplest useful implementation is a human-readable markdown file. This
 * example ships {@link MarkdownMemoryProvider} (in `memory-provider.ts`),
 * composes it into a {@link storage.StorageProvider} (NoOp for the other three
 * domains), and runs the SAME agent twice against it:
 *
 * - `--phase store`  — the agent is given facts about a fictional "Project
 *   Ironwood" and writes each as a memory via the built-in `memory` tool. The
 *   process exits leaving a readable `memory.md` on disk.
 * - `--phase recall` — a fresh process loads `memory.md` through the same
 *   provider and answers questions that restate NONE of the facts. The agent
 *   recalls them from memory via the `memory` tool.
 *
 * ## The seam
 * `main` never calls `appendMemory`/`getMemories` directly. It hands the
 * composed provider to `HarnessBuilder.storage(...)`; the harness threads
 * `storage.memory()` into the built-in `memory` tool's `ToolContext` per run.
 * The agent drives all reads/writes from inside the ReAct loop. Swap the
 * provider (e.g. the built-in JSONL `FileSystemStorageProvider`) and nothing
 * else changes — that is the point of the seam.
 *
 * ## Pinned session id (critical)
 * Memory is keyed by `SessionId`; the `memory` tool always uses `ctx.sessionId`.
 * Both phases therefore pin the SAME id — `SESSION` =
 * `SessionId.of("project-ironwood")`, NOT `SessionId.generate()`. With a
 * generated id Run 2 would key a different session and read nothing back.
 *
 * ## Scope
 * All facts use `StorageScope` `"project"` (the `memory` tool rejects
 * `"local"`). The prompts instruct the agent to use `scope: "project"`
 * consistently so the recall read hits the same scope the store writes wrote.
 *
 * There are no SPEC QUESTION markers in this file.
 *
 * ## Run it
 *
 * ```sh
 * ollama serve &
 * ollama pull llama3.2
 * pnpm install
 * pnpm start -- --phase store     # writes memory.md
 * cat memory.md                   # inspect the human-readable artifact
 * pnpm start -- --phase recall    # answers from memory.md alone
 * ```
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  HarnessBuilder,
  OLLAMA_DEFAULT_BASE_URL,
  OllamaModelInterface,
  SessionId,
  newTask,
  type HarnessStreamEvent,
} from "@spore/core";
import { StandardTools } from "@spore/tools";

import { MarkdownMemoryProvider } from "./memory-provider.js";

/**
 * The pinned session id shared by BOTH phases. Memory is session-keyed, so store
 * and recall MUST agree on this id or recall reads nothing. NOT
 * `SessionId.generate()`.
 */
const SESSION = SessionId.of("project-ironwood");

const STORE_SYSTEM_PROMPT =
  "You are a memory-keeping agent. You will be given a briefing of facts. For " +
  'EACH distinct fact, call the `memory` tool with operation "write", scope ' +
  '"project", role "assistant", and the fact text as `content`. Write the facts ' +
  "verbatim and one at a time. Do not summarize or merge facts. When every fact " +
  "has been written, reply with a short confirmation of how many you stored.";

const RECALL_SYSTEM_PROMPT =
  "You are a recall agent. Everything you know about Project Ironwood lives in " +
  "memory — nothing is in this prompt. FIRST call the `memory` tool with " +
  'operation "read", scope "project" to load what you remember. THEN answer the ' +
  "user's questions using only the recalled memories. Cite the relevant " +
  "remembered fact when you answer. Do not invent facts that are not in memory.";

const RECALL_QUESTIONS =
  "Answer these about Project Ironwood, using only your memory:\n" +
  "1. How many engineers are on the team, and who leads it?\n" +
  "2. What database was chosen as the system of record, and why over the alternative?\n" +
  "3. What are the two hard constraints?\n" +
  "4. What is the known single point of failure?";

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const modelId =
    argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "llama3.2";
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;

  // `memory.md` lives next to this example's sources so `pnpm start` works from
  // anywhere and the artifact is easy to find and inspect between phases.
  const here = dirname(fileURLToPath(import.meta.url));
  const exampleRoot = join(here, "..");
  const memoryPath = join(exampleRoot, "memory.md");

  // Default (no --phase): run store, then point the user at recall and exit.
  const phase = argValue(args, "--phase") ?? "store";
  const defaultPhase = argValue(args, "--phase") === undefined;

  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);

  console.log(`model      : ${modelId}`);
  console.log(`memory.md  : ${memoryPath}`);
  console.log(
    `session id : ${SESSION.asString()}  (pinned — shared by both phases)`,
  );
  console.log(`phase      : ${phase}\n`);

  if (phase === "store") {
    // Read the briefing and feed it to the agent. The agent writes each fact via
    // the `memory` tool; `main` never writes memory itself. Re-running store just
    // appends — delete memory.md by hand to reset.
    const briefing = readFileSync(
      join(exampleRoot, "project-ironwood.md"),
      "utf8",
    );
    const prompt =
      "Here is the Project Ironwood briefing. Store each fact to memory.\n\n" +
      briefing;

    const output = await runPhase(
      model,
      memoryPath,
      STORE_SYSTEM_PROMPT,
      prompt,
      baseUrl,
    );
    console.log(`\nstored. agent said: ${output}`);
    if (existsSync(memoryPath)) {
      console.log(`\nmemory.md now exists on disk: ${memoryPath}`);
      console.log("inspect it, then run:  pnpm start -- --phase recall");
    }
    if (defaultPhase) {
      console.log(
        "\n(no --phase given, so we ran `store`. Now run `pnpm start -- --phase recall`.)",
      );
    }
    return;
  }

  if (phase === "recall") {
    if (!existsSync(memoryPath)) {
      console.error(
        `memory.md does not exist yet at ${memoryPath}.\n` +
          "Run `pnpm start -- --phase store` first.",
      );
      process.exit(2);
    }
    const output = await runPhase(
      model,
      memoryPath,
      RECALL_SYSTEM_PROMPT,
      RECALL_QUESTIONS,
      baseUrl,
    );
    console.log(`\nanswers from memory:\n${output}`);
    return;
  }

  console.error(
    `unknown --phase ${JSON.stringify(phase)}. Use \`store\` or \`recall\`.`,
  );
  process.exit(2);
}

/**
 * Build a harness over the markdown memory provider + the built-in `memory`
 * tool, pin the shared session id, run one task, and stream the loop.
 */
async function runPhase(
  model: OllamaModelInterface,
  memoryPath: string,
  systemPrompt: string,
  taskPrompt: string,
  baseUrl: string,
): Promise<string> {
  // Compose the real markdown MemoryStore with NoOp for the other three storage
  // domains. This is the entire integration: the harness threads
  // `storage.memory()` into the `memory` tool's context per run.
  const storage = new MarkdownMemoryProvider(memoryPath).intoStorageProvider();

  const harness = HarnessBuilder.conversational(model)
    .storage(storage) // ← the seam
    .tool(StandardTools.memory()) // ← the built-in memory read/write tool
    .systemPrompt(systemPrompt)
    // Structured mode helps small Ollama models emit clean tool calls (one per
    // turn, no interleaved reasoning — so the "think · turn N" line is just a
    // turn marker, not model chatter).
    .modelParams({ structured_tool_calls: true, stop_sequences: [] })
    .build();

  // PIN the session id — both phases pass the same one so recall reads what
  // store wrote.
  const task = newTask(taskPrompt, SESSION, {
    kind: "re_act",
    max_iterations: 20,
  });

  const onStream = (event: HarnessStreamEvent): void => {
    switch (event.kind) {
      case "turn_start":
        console.log(`think  · turn ${event.turn}`);
        break;
      case "tool_call":
        console.log(
          `    act    → ${event.name}(${truncate(JSON.stringify(event.args ?? {}), 160)})`,
        );
        break;
      case "tool_result": {
        const tag = event.is_error ? "obs(err)" : "obs ";
        console.log(`    ${tag}→ ${truncate(event.content ?? "", 160)}`);
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
      `\ncould not reach the model — is Ollama running at ${baseUrl}? (\`ollama serve\`)\n` +
        `${err instanceof Error ? err.message : String(err)}`,
    );
    process.exit(1);
  }

  if (result.kind === "success") return result.output;
  console.error(`\nrun did not succeed: ${JSON.stringify(result)}`);
  process.exit(1);
}

function argValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

/** Keep stream lines readable — memory reads return a JSON array of entries. */
function truncate(s: string, max: number): string {
  const flat = s.replace(/\n/g, " ");
  const chars = [...flat];
  return chars.length <= max ? flat : `${chars.slice(0, max).join("")}…`;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
