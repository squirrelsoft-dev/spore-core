/**
 * spore-core example 09 — the `SelfVerifying` loop strategy.
 *
 * ## What this example demonstrates
 *
 * **Quality loops are a harness concern, not application logic.** The agent
 * drafts a TypeScript function, a *fresh evaluator run* critiques that draft
 * against an explicit spec, and a {@link verifier.Verifier} turns the critique
 * into a verdict. If the verdict is FAIL, the reason is injected back into the
 * build context and the loop revises. This repeats until the verifier returns
 * `passed` or `max_iterations` is exhausted. You write **no loop code** — you
 * wire a composed strategy (`SelfVerifying(inner: ReAct{worker-schema}, evaluator)`)
 * and a `Verifier`, and the harness runs the verify→revise cycle for you.
 *
 * Post-#119 the strategy is a composed tree: `SelfVerifying`'s `inner` (worker)
 * slot is STRUCTURED, so a bare `ReAct` there MUST declare an `output` schema
 * (here `worker-schema`), which the {@link ExecutionRegistry} validates at run
 * entry. The `evaluator` stays the EMPTY handle (`""`), default-filled from the
 * single-collaborator `.verifier(..)` setter (the pre-#119 `.evaluatorAgent(..)`
 * seam was removed in #124 — the evaluate phase now defaults to the worker's
 * agent). The only handle registered explicitly is `worker-schema`.
 *
 * ## The task the agent-under-test must solve
 *
 * Write a TypeScript function `parseIntList(input: string): ParseIntListResult`
 * that parses a comma-separated list of integers. `ParseIntListResult` is an
 * idiomatic discriminated result —
 * `{ ok: true; value: number[] } | { ok: false; error: ParseIntListError }` —
 * so failure is a typed value, never an unexpected throw. The verifier checks
 * **five** criteria, each explicitly:
 *
 * 1. **Signature**: takes a `string`, returns the discriminated
 *    `ParseIntListResult` union with a custom `ParseIntListError` type defined
 *    in the same file.
 * 2. **Edge cases**: empty/whitespace-only input → `{ ok: true, value: [] }`;
 *    whitespace around each number tolerated (`" 1, 2 ,3 "`); a non-integer
 *    token → `{ ok: false, error }`, never an unexpected throw/crash.
 * 3. **Doc comments**: a JSDoc (`/** … *​/`) block on the function.
 * 4. **No crashers**: no non-null assertions (`!`) or unchecked casts that can
 *    crash; the function returns the error variant instead.
 * 5. **At least one usage example** in the JSDoc (an `@example` block showing a
 *    call and its result).
 *
 * ## How the draft reaches the evaluator — and why we need a file tool
 *
 * Reading the strategy source (`runSelfVerifying` / `runEvaluatePhase` in
 * `packages/core/src/harness/standard.ts`) settles the tool question. The
 * evaluate phase builds a **fresh** evaluator run whose context is seeded ONLY
 * with a directive containing the task instruction plus a read-only sandbox.
 * The build agent's draft text is **not** auto-injected into the evaluator's
 * context. So for the evaluator to actually read the draft, the draft must live
 * on disk where the (read-only) evaluator can read it.
 *
 * Therefore this example wires exactly the minimal file tool set:
 * - `write_file` — the **build** agent saves its draft to
 *   `workspace/parse-int-list.ts`.
 * - `read_file`  — the **evaluator** reads that file back (its `write_file` is
 *   blocked by the internally-derived read-only sandbox).
 *
 * No `web_search`, no shell, nothing else. The loop is the point.
 *
 * ## The observability seam — `ReportingVerifier`
 *
 * Sub-loop streaming is suppressed by design (the build and evaluate sub-runs
 * run with a suppressed sink, exactly like PlanExecute). The ONE reliable seam
 * to watch the verify→revise cycle is the {@link verifier.Verifier} itself: the
 * harness calls `verify(input)` once per iteration, and {@link
 * verifier.VerifierInput} carries the **draft** (`build_result` output), the
 * **critique** (`eval_result` output), and the 0-indexed `iteration`. So we
 * wrap {@link verifier.EvaluatorResponseVerifier} in a small `ReportingVerifier`
 * that prints, each iteration: a 1-based header with the configured max, the
 * draft, the critique, and the verdict — then delegates the actual pass/fail
 * decision to the inner verifier.
 *
 * `EvaluatorResponseVerifier` matches the evaluator's text against a `PASS`
 * pattern and a `FAIL: <reason>` pattern; if NEITHER matches it returns FAIL by
 * contract (default-to-FAIL is baked into the verifier and reinforced by the
 * harness's evaluator directive — "you did NOT write this code; default to FAIL
 * unless you can confirm it is right").
 *
 * ## Run it
 *
 * ```sh
 * ollama serve &
 * ollama pull llama3.2
 * pnpm install
 * pnpm start                              # llama3.2, 3 iterations
 * pnpm start -- --max-iterations 5
 * pnpm start -- --model qwen2.5-coder:7b
 * ```
 *
 * See the README for the honest rough-edges section: SelfVerifying against a
 * small local model is genuinely flaky (the evaluator may mis-judge, the loop
 * may exhaust without passing). A larger hosted model gives a cleaner demo.
 */

import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  ExecutionRegistry,
  HarnessBuilder,
  OLLAMA_DEFAULT_BASE_URL,
  OllamaModelInterface,
  SessionId,
  WorkspaceScopedSandbox,
  newTask,
  reactPerLoop,
  verifier,
  type LoopStrategy,
  type RunResult,
} from "@spore/core";
import { StandardTools } from "@spore/tools";

/** The file the build agent writes and the evaluator reads back. */
const DRAFT_FILENAME = "parse-int-list.ts";

/**
 * The spec the agent must satisfy. It is the `Task` instruction, so the **build**
 * agent sees it directly, and — because the evaluate phase embeds the task
 * instruction in the evaluator's directive — the **evaluator** sees the exact
 * same five criteria. One source of truth for both roles.
 */
const TASK_PROMPT = `Write a TypeScript function named \`parseIntList\` and save it to the file \
\`${DRAFT_FILENAME}\` using the write_file tool. It must satisfy ALL of the \
following, which you will be graded on criterion-by-criterion:

1. SIGNATURE: \`export function parseIntList(input: string): ParseIntListResult\` \
where \`ParseIntListResult\` is an idiomatic discriminated union \
\`{ ok: true; value: number[] } | { ok: false; error: ParseIntListError }\` and \
\`ParseIntListError\` is a custom error type you define in the same file.
2. EDGE CASES: empty or whitespace-only input returns \`{ ok: true, value: [] }\`; \
whitespace around each number is tolerated (e.g. " 1, 2 ,3 " parses to [1, 2, 3]); \
a non-integer token returns the \`{ ok: false, error }\` variant and NEVER throws \
an unexpected error or crashes.
3. DOC COMMENTS: the function has a JSDoc (/** ... */) block describing what it does.
4. NO CRASHERS: no non-null assertions (the \`!\` postfix operator) and no unchecked \
casts that can crash; on bad input return the error variant instead.
5. USAGE EXAMPLE: include at least one usage example in the JSDoc — an \`@example\` \
block showing a call and the result it returns.

Write ONLY the file contents (valid TypeScript). Save it with write_file, then \
report that you are done.`;

/**
 * System prompt shared by the build agent and the evaluator agent (the harness
 * system prompt is shared across both phases). It is deliberately role-neutral:
 * the build/evaluate framing is supplied per-phase (the build agent gets the
 * spec as its task; the evaluator gets the harness's built-in review directive
 * plus the same spec). It reinforces the file-tool contract and the evaluator's
 * default-to-FAIL posture.
 */
const SYSTEM_PROMPT = `You work on TypeScript code. Your only tools are write_file (save a file to the \
workspace) and read_file (read a file back). You have no shell and cannot run or \
compile code.

When ASKED TO WRITE code: write the file with write_file, then say you are done.

When ASKED TO REVIEW code: first read_file the file under review. Then check the \
work against EACH numbered criterion in the task, one at a time. You did NOT \
write this code — default to FAIL unless you can positively confirm every \
criterion holds. Respond with EXACTLY ONE verdict line as the LAST line of your \
reply:
  - \`PASS\` if and only if every criterion holds, or
  - \`FAIL: <which criteria failed and why>\` otherwise.
Never emit PASS when unsure.`;

/**
 * A {@link verifier.Verifier} decorator: prints the verify→revise cycle to
 * stdout, then delegates the actual verdict to an inner verifier.
 *
 * This is the one reliable observability seam for SelfVerifying — the build and
 * evaluate sub-runs are streamed with a suppressed sink, so the verifier call is
 * where the draft + critique + verdict become visible. Per iteration it prints:
 * a 1-based header with the configured max, the **draft** (`build_result`
 * output), the **critique** (`eval_result` output), and the **verdict**.
 */
class ReportingVerifier implements verifier.Verifier {
  constructor(
    private readonly inner: verifier.Verifier,
    private readonly max: number,
  ) {}

  async verify(
    input: verifier.VerifierInput,
    signal?: AbortSignal,
  ): Promise<verifier.VerifierVerdict> {
    // `iteration` is 0-indexed on the wire; display it 1-based.
    const n = input.iteration + 1;
    console.log(`\n══════════════ iteration ${n}/${this.max} ══════════════`);

    console.log("\n── draft (what the agent wrote) ──");
    console.log(runResultOutput(input.build_result));

    console.log("\n── evaluation (the critique) ──");
    console.log(runResultOutput(input.eval_result));

    // Delegate the actual decision to the inner verifier.
    const verdict = await this.inner.verify(input, signal);

    console.log("\n── verdict ──");
    if (verdict.kind === "passed") {
      console.log("PASS — criteria satisfied; loop halts.");
    } else {
      console.log(`FAIL — ${verdict.reason}`);
      if (n < this.max) {
        console.log("(reason injected into next build turn; revising…)");
      } else {
        console.log("(no iterations left; loop will exhaust)");
      }
    }
    console.log("════════════════════════════════════════════════");
    return verdict;
  }

  maxIterations(): number {
    return this.max;
  }
}

/**
 * Reduce a {@link RunResult} to printable text: the `success` output, or a short
 * description of why the run did not complete.
 */
function runResultOutput(r: RunResult): string {
  switch (r.kind) {
    case "success":
      return r.output;
    case "failure":
      return `<run did not complete: ${JSON.stringify(r.reason)}>`;
    case "waiting_for_human":
      return "<run paused waiting for human>";
    case "escalate":
      return `<run escalated: ${r.signal.kind}>`;
    // This example never wires a consult ladder; a top-level consult is not
    // expected, but the variant must be handled so the switch stays exhaustive.
    case "consult":
      return `<run paused on consult: ${r.request.kind}>`;
  }
}

/**
 * The worker (build-phase) output contract (`worker-schema`). Post-#119,
 * `SelfVerifying`'s `inner` (worker) slot is STRUCTURED: a bare `ReAct` there
 * must declare an `output` schema so its result is EVALUABLE
 * ({@link ExecutionRegistry.validate} enforces this via its structured-slot
 * check). The build agent writes the draft file; this advertises the path it wrote.
 */
function workerSchema(): unknown {
  return {
    type: "object",
    properties: {
      file: { type: "string", description: "Path the draft was written to." },
      summary: { type: "string", description: "What was implemented." },
    },
    required: ["file"],
  };
}

/**
 * The {@link ExecutionRegistry} the composed strategy's handles resolve against.
 * Only `worker-schema` is EXPLICIT; the builder default-fills the empty agent /
 * toolset handles (`reactPerLoop`) AND the empty-key evaluator from `.verifier(..)`
 * at `build`. So the SelfVerifying `evaluator` stays the EMPTY handle (`""`).
 */
export function buildRegistry(): ExecutionRegistry {
  return ExecutionRegistry.builder()
    .schema("worker-schema", workerSchema())
    .build();
}

/**
 * The post-#119 composed strategy: `SelfVerifying(inner: ReAct, evaluator)`. The
 * worker leaf carries the `worker-schema` output contract (required for the
 * structured `worker` slot) and a `per_loop(maxIterations)` build budget. The
 * `evaluator` is the EMPTY handle (`""`), which the builder default-fills from
 * `.verifier(..)` (#124 single-collaborator migration seam). Old flat shape was
 * `{ kind: "self_verifying" }`.
 */
export function selfVerifyingStrategy(maxIterations: number): LoopStrategy {
  return {
    kind: "self_verifying",
    inner: { ...reactPerLoop(maxIterations), output: "worker-schema" },
    evaluator: "",
  };
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  const modelId =
    argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "llama3.2";
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;

  // Max iterations: CLI flag wins, then env var, then default 3.
  const maxIterations = parsePositiveInt(
    argValue(args, "--max-iterations") ?? process.env.SPORE_MAX_ITERATIONS,
    3,
  );

  const prompt = argValue(args, "--prompt") ?? TASK_PROMPT;

  // The agents operate inside this example's `workspace/` directory. Resolve it
  // relative to this source file so `pnpm start` works from anywhere, and create
  // it if missing — the sandbox requires an existing root, which it canonicalizes
  // at construction.
  const here = dirname(fileURLToPath(import.meta.url));
  const workspaceRoot = join(here, "..", "workspace");
  mkdirSync(workspaceRoot, { recursive: true });
  const draftPath = join(workspaceRoot, DRAFT_FILENAME);

  // Post-#119/#124 the `.evaluatorAgent(..)` single-collaborator seam is GONE.
  // The evaluate phase now defaults to the inner worker's agent (the empty-key
  // agent the builder default-fills from `conversational(model)`), running a
  // FRESH evaluator turn for which the harness internally derives a read-only
  // sandbox — so its `write_file` is blocked but `read_file` works, exactly what a
  // reviewer needs. The judging seam is the verifier below (the SelfVerifying
  // `evaluator` empty handle resolves to it via the empty-key default fill).

  // The verifier: pattern-match the evaluator's text. `PASS` (anchored,
  // case-insensitive, multiline) → passed; `FAIL: <reason>` → failed(reason);
  // neither → failed by contract (default-to-FAIL). JS has no inline `(?im)`
  // flags, but `EvaluatorResponseVerifier` accepts the Rust fixture form and
  // strips the leading inline-flag group, so we pass the same patterns the other
  // language ports use. Wrapped in `ReportingVerifier` so the cycle is visible;
  // the harness reads `maxIterations()` off the OUTER verifier, so both carry the
  // same cap.
  const innerVerifier = new verifier.EvaluatorResponseVerifier({
    pass_pattern: "(?im)^\\s*PASS\\s*$",
    fail_pattern: "(?im)FAIL:\\s*.+",
    max_iterations: maxIterations,
  });
  const reportingVerifier = new ReportingVerifier(innerVerifier, maxIterations);

  // Build harness: conversational preset, workspace sandbox, the minimal file
  // tool set (write_file for the builder + read_file for the evaluator), shared
  // system prompt, the registry (carrying the `worker-schema` output contract),
  // and the verifier (folded into the default-key evaluator handle).
  const buildModel = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const sandbox = new WorkspaceScopedSandbox({ root: workspaceRoot });
  const harness = HarnessBuilder.conversational(buildModel)
    .sandbox(sandbox)
    .registry(buildRegistry())
    .tool(StandardTools.writeFile())
    .tool(StandardTools.readFile())
    .systemPrompt(SYSTEM_PROMPT)
    .verifier(reportingVerifier)
    .build();

  // THE STRATEGY. There is no loop code below — the harness runs the
  // verify→revise cycle. The worker leaf carries the `worker-schema` output
  // contract (required for the structured `worker` slot) and a
  // `per_loop(maxIterations)` build budget; the empty `evaluator` handle resolves
  // to the verifier above. A generous `max_turns` per build/evaluate sub-run lets
  // a small model take a few tool calls before claiming done.
  const task = newTask(
    prompt,
    SessionId.generate(),
    selfVerifyingStrategy(maxIterations),
    { max_turns: 12 },
  );

  console.log(`model         : ${modelId}`);
  console.log(`base url      : ${baseUrl}`);
  console.log(`workspace     : ${workspaceRoot}`);
  console.log("strategy      : SelfVerifying (draft → critique → revise)");
  console.log(`max iterations: ${maxIterations}`);
  console.log(
    "verifier      : EvaluatorResponseVerifier (PASS / FAIL:) wrapped in ReportingVerifier",
  );
  console.log(
    "\nThe agent will draft `parseIntList`, an evaluator will critique it against the",
  );
  console.log(
    `five spec criteria, and the loop revises until PASS or ${maxIterations} iteration(s) elapse.\n`,
  );

  let result: RunResult;
  try {
    result = await harness.run({ task });
  } catch (err) {
    console.error(
      `\ncould not reach the model — is Ollama running at ${baseUrl}? (\`ollama serve\`)\n${err instanceof Error ? err.message : String(err)}`,
    );
    process.exit(1);
  }

  if (result.kind === "success") {
    console.log(
      `\n✓ PASSED — the evaluator accepted the draft (after at most ${maxIterations} iteration(s), ${result.turns} build turn(s) total).`,
    );
    if (existsSync(draftPath)) {
      console.log(
        `\n── final function (${draftPath}) ──\n${readFileSync(draftPath, "utf8")}`,
      );
    }
    return;
  }

  if (
    result.kind === "failure" &&
    result.reason.kind === "self_verify_exhausted"
  ) {
    console.log(
      `\n✗ EXHAUSTED — ${result.reason.iterations} iteration(s) elapsed without a PASS.`,
    );
    console.log(`last failure reason: ${result.reason.last_reason}`);
    if (existsSync(draftPath)) {
      console.log(
        `\n── last draft on disk (${draftPath}) ──\n${readFileSync(draftPath, "utf8")}`,
      );
    }
    console.log(
      "\nThis is an expected rough edge with small local models — see the README. " +
        "Try a larger model or raise --max-iterations.",
    );
    process.exit(1);
  }

  console.error(`\nrun did not succeed: ${JSON.stringify(result)}`);
  process.exit(1);
}

function argValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

/** Parse a positive integer, falling back to `fallback` on absent/invalid/non-positive. */
function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  const n = Number.parseInt(value, 10);
  return Number.isInteger(n) && n > 0 ? n : fallback;
}

/** Run `main` only when this module is the program entrypoint — NOT when it is
 *  imported (e.g. by the composition test, which reuses `buildRegistry` /
 *  `selfVerifyingStrategy`). */
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
