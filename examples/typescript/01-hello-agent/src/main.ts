/**
 * spore-core example 01 — hello agent.
 *
 * The smallest real thing you can build with spore-core: turn a model into a
 * running agent and ask it to say hello. No tools, no filesystem, no
 * multi-turn state.
 *
 * `HarnessBuilder.conversational(model).build()` defaults every required
 * component (a model-backed agent, an empty tool registry, a null sandbox, a
 * standard context manager, and respond-and-stop termination), so the whole
 * thing is a few lines. Later examples override individual defaults — add
 * tools, swap the sandbox, change the loop strategy — via the builder setters.
 *
 * ## Run it
 *
 * ```sh
 * ollama serve &            # start a local model server
 * ollama pull llama3.2      # pull the default model
 * pnpm install              # link @spore/core (file: dependency)
 * pnpm start                # or: pnpm start -- --model <id>
 * ```
 *
 * `SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and the
 * Ollama endpoint (default `http://localhost:11434`).
 */

import {
  HarnessBuilder,
  OLLAMA_DEFAULT_BASE_URL,
  OllamaModelInterface,
  simpleTask,
} from "@spore/core";

async function main(): Promise<void> {
  // Model id + endpoint come from args/env so you can swap models without an
  // edit.
  const args = process.argv.slice(2);
  const modelId = argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "llama3.2";
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;

  // A model, a harness, a task — that's the whole setup.
  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const harness = HarnessBuilder.conversational(model).build();
  const task = simpleTask("Reply with a friendly one-line greeting.");

  console.log(`model      : ${modelId}`);
  const result = await harness.run({ task });
  if (result.kind === "success") {
    console.log(`result     : Success (${result.turns} turn(s))`);
    console.log(`greeting   : ${result.output}`);
    return;
  }
  console.error(`result     : ${JSON.stringify(result)}`);
  process.exit(1);
}

function argValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
