/**
 * spore-core example 02 — conversational REPL.
 *
 * Takes example 01 one step further: an interactive chat loop where the agent
 * remembers what you said earlier in the session. Same
 * `HarnessBuilder.conversational(model)` harness as 01 — the new idea here is
 * *conversation continuity across runs*.
 *
 * ## How memory works
 *
 * The harness is stateless between `run()` calls: each call takes an optional
 * starting `SessionState` (the message history) and drives one task to a final
 * response. As of issue #102, `RunResult` (success) now hands the post-run
 * `SessionState` back, so the caller resumes the conversation LOSSLESSLY — no
 * reconstruction. After each turn we feed the returned `session_state` straight
 * into the next run via `HarnessRunOptions.session_state`. The harness appends
 * the new user line on top of that history before calling the model, so the
 * model sees the whole conversation and can refer back to it.
 *
 * This works for tool-using agents too: the returned `session_state` carries the
 * tool-call and tool-result messages the loop produced, which the old
 * "reconstruct history from `output`" trick could not recover.
 *
 * Prefer it hands-free? Wire a `StorageProvider` and call
 * `HarnessBuilder.autoPersistSessions(true)`: the harness then auto-loads and
 * auto-persists by `session_id`, so you reuse the id instead of threading state
 * at all (great for a service that resumes across restarts).
 *
 * ## Run it
 *
 * ```sh
 * ollama serve &
 * ollama pull llama3.2
 * pnpm install
 * pnpm start                # then chat; /exit or Ctrl-D to quit
 * ```
 */

import { createInterface } from "node:readline";

import {
  HarnessBuilder,
  OLLAMA_DEFAULT_BASE_URL,
  OllamaModelInterface,
  SessionId,
  emptySessionState,
  newTask,
  reactPerLoop,
  runResultSessionState,
  type SessionState,
} from "@spore/core";

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const modelId = argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "llama3.2";
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;

  // Build the harness once; reuse it for every turn.
  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const harness = HarnessBuilder.conversational(model).build();

  // One session id for the whole REPL, and the conversation state we thread back
  // in on each turn. Each run hands the post-run `SessionState` back (issue
  // #102), so we just carry it forward — lossless, no reconstruction.
  const sessionId = SessionId.generate();
  let state: SessionState = emptySessionState();
  let turnsExchanged = 0;

  console.log(
    `conversational REPL — model ${modelId}. Type a message; /exit or Ctrl-D to quit.`,
  );

  const rl = createInterface({ input: process.stdin });
  process.stdout.write("you> ");
  for await (const raw of rl) {
    const line = raw.trim();
    if (line.length === 0) {
      process.stdout.write("you> ");
      continue;
    }
    if (line === "/exit" || line === "/quit") break;

    // Thread the running state into this turn. The harness appends `line` as the
    // new user message before calling the model.
    const task = newTask(line, sessionId, reactPerLoop(4));
    const result = await harness.run({ task, session_state: state });

    if (result.kind === "success") {
      console.log(`bot> ${result.output}`);
      // Carry the post-run state forward losslessly (issue #102): it already
      // contains this turn's user + assistant messages (and any tool messages a
      // tool-using agent would produce).
      state = runResultSessionState(result);
      turnsExchanged += 1;
    } else {
      console.error(`bot> [run did not succeed: ${JSON.stringify(result)}]`);
    }
    process.stdout.write("you> ");
  }
  rl.close();
  console.log();

  console.log(`bye (${turnsExchanged} turn(s); ${state.messages.length} message(s) in history)`);
}

function argValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
