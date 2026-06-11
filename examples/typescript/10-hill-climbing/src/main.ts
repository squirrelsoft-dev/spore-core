/**
 * spore-core example 10 — the `HillClimbing` loop strategy.
 *
 * ## What this example demonstrates
 *
 * **Iterative refinement under a scoring oracle is a harness concern, not
 * application logic.** The agent edits ONE file in place
 * (`workspace/README.md`) across iterations. After every iteration a custom
 * {@link metric.MetricEvaluator} reads that file and asks a *separate judge
 * model* to score it on three dimensions — Clarity, Completeness, Example
 * quality (0–10 each) — returning the total/30 normalized to `[0,1]`. The
 * harness applies its keep-or-revert rule ({@link metric.shouldKeep}): a
 * *strictly* better score is KEPT; anything else is DISCARDED, and because
 * `revert_on_no_improvement` is on, the workspace is `git reset --hard`-ed back
 * to the best-so-far. The loop halts on **stagnation** (`MAX_STAGNATION`
 * consecutive non-improvements) or **budget** (`max_turns`). You write **no
 * loop code** — you wire a strategy, a metric evaluator, and an observability
 * sink, and the harness runs the climb.
 *
 * ## The contrast with example 09 (SelfVerifying) — the teaching point
 *
 * 09 has a **binary exit condition**: a `Verifier` returns PASS and the loop
 * *succeeds*, or it exhausts and *fails*. HillClimbing has **no PASS**. It is
 * an optimization loop: there is only *best-so-far*. It does not know it is
 * "done" — it only knows it has stopped improving. The terminal outcome is
 * therefore a `stagnation_limit_reached` or `budget_exceeded` halt, NOT a
 * success/fail verdict on quality.
 *
 * ## SPEC NOTE — why this diverges from issue #99's original framing (Option A)
 *
 * The original issue asked the agent to "climb until total ≥ 25/30 or max
 * iterations". Planning (#99 spec-resolution comment) established that framing
 * does NOT match the real `HillClimbing` strategy in spore-core:
 *   - There is no score-threshold success condition. The loop keeps/reverts on
 *     *relative* improvement and halts on stagnation/budget — it never compares
 *     the metric against an absolute target.
 *   - `MAX_ITERATIONS` is not a HillClimbing parameter; iterations are bounded
 *     by `BudgetLimits.max_turns`. The `MAX_ITERATIONS` constant maps there.
 *   - The shipped `LlmJudgeEvaluator` scores a FIXED construction-time string,
 *     so it cannot see the evolving draft. This example therefore ships a small
 *     example-local {@link ReadmeQualityEvaluator} that reads `workspace/
 *     README.md` through the sandbox each iteration before scoring.
 *
 * Resolution = **Option A** (reframe to real semantics, no core change):
 *   - {@link SCORE_THRESHOLD} (25/30) is kept as a **DISPLAY annotation only**.
 *     When a draft's total crosses it, the printed line is marked `★ crossed
 *     target threshold`. It does **not** terminate the loop.
 *   - The per-iteration print is split across two seams, mirroring how the
 *     harness actually exposes the run:
 *       * the evaluator prints the draft + 3 sub-scores + total (it is the only
 *         place that sees the rubric breakdown), and
 *       * a custom {@link observability.ObservabilityProvider} handling the
 *         `hill_climbing_iteration` warn event prints the kept/discarded/
 *         reverted decision (iteration, metric value, delta) — the harness
 *         emits exactly one such event per iteration.
 *
 * ## Wiring note — the post-#119 composed strategy + registry
 *
 * Post-#119 the strategy is a composed tree:
 * `HillClimbing(inner: ReAct{propose-schema}, evaluator)`. Its `inner` (propose)
 * slot is STRUCTURED, so a bare `ReAct` there MUST declare an `output` schema
 * (here `propose-schema`), which the {@link ExecutionRegistry} validates at run
 * entry. The `evaluator` stays the EMPTY handle (`""`), default-filled from the
 * fluent `.metricEvaluator(...)` setter (#124) — so the only handle registered
 * explicitly is `propose-schema`. The observability sink is wired via
 * `.observability(...)`. `max_stagnation` / `min_improvement_delta` are now bare
 * values (no longer `Option`-wrapped), and the direction enum is
 * `HillClimbingDirection` (renamed from `OptimizationDirection`).
 *
 * ## Constants (see their doc comments below)
 *   - {@link MAX_ITERATIONS}  — maps to `BudgetLimits.max_turns` (default 6).
 *   - {@link MAX_STAGNATION}  — consecutive non-improvements before halt (2).
 *   - {@link SCORE_THRESHOLD} — DISPLAY annotation only (25). Never terminates.
 *   - {@link DIMENSION_MAX} / {@link TOTAL_MAX} — 10 per dimension, 30 total.
 *
 * ## The seams this example wires
 *   - {@link ReadmeQualityEvaluator}   — `implements MetricEvaluator`; reads the
 *     file via the sandbox, runs a fresh judge model call, prints the rubric.
 *   - {@link ReportingObservability}   — extends `InMemoryObservabilityProvider`
 *     and overrides `emitWarn` to print each `hill_climbing_iteration` decision.
 *
 * ## Run it
 *
 * ```sh
 * ollama serve &
 * ollama pull llama3.2
 * pnpm install
 * pnpm start                              # default model llama3.2, 6-iteration budget
 * pnpm start -- --max-iterations 8
 * pnpm start -- --model qwen2.5-coder:7b
 * ```
 *
 * See the README for the honest rough-edges section.
 */

import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  ExecutionRegistry,
  OLLAMA_DEFAULT_BASE_URL,
  OllamaModelInterface,
  SessionId,
  WorkspaceScopedSandbox,
  HarnessBuilder,
  newTask,
  metric,
  observability,
  reactPerLoop,
  termination,
  type HillClimbingDirection,
  type LoopStrategy,
  type ModelInterface,
  type ModelRequest,
  type RunResult,
  type SandboxProvider,
} from "@spore/core";
import { StandardTools } from "@spore/tools";

// ============================================================================
// Constants — all clearly named, with the spec semantics in their doc comments.
// ============================================================================

/**
 * Climbing-iteration ceiling. Maps to `BudgetLimits.max_turns` — this is the
 * BUDGET, not a success target. The loop may halt EARLIER on stagnation. There
 * is no "reached the goal" outcome; HillClimbing always halts on budget or
 * stagnation. Six gives a small local model room to make a few real edits.
 */
const MAX_ITERATIONS = 6;

/**
 * Consecutive non-improvements tolerated before the loop halts with
 * `stagnation_limit_reached`. The stagnation counter resets to 0 on any kept
 * (strictly-improving) iteration. Maps to `max_stagnation`.
 */
const MAX_STAGNATION = 2;

/**
 * Per-iteration build budget for the propose leaf. Post-#119, HillClimbing's
 * `inner` is a composed `ReAct(reactPerLoop(PER_ITER_BUDGET))` — this bounds the
 * build agent's tool calls WITHIN one climb iteration. The NUMBER of iterations
 * is bounded separately by `max_turns`.
 */
const PER_ITER_BUDGET = 8;

/**
 * DISPLAY ANNOTATION ONLY. When a draft's total score (0–30) reaches this, the
 * evaluator marks the printed line `★ crossed target threshold`. SPEC NOTE:
 * this does NOT terminate the loop — HillClimbing has no score-threshold exit.
 */
const SCORE_THRESHOLD = 25;

/** Max score per rubric dimension (Clarity, Completeness, Example quality). */
const DIMENSION_MAX = 10;

/** Max total across the three dimensions (`3 * DIMENSION_MAX`). */
const TOTAL_MAX = 3 * DIMENSION_MAX;

/**
 * The file under refinement, relative to the workspace root. The build agent
 * edits this in place; the evaluator reads it back through the sandbox.
 */
const DRAFT_FILENAME = "README.md";

/**
 * The task the build agent is asked to perform each iteration. It edits ONE
 * file in place — the climb is over successive revisions of the same README.
 */
const TASK_PROMPT = `You are writing the README.md for a fictional Rust crate called \`ironwood\`, a \
small library for parsing and validating semantic-version strings. Use the \
write_file tool to save your README to \`${DRAFT_FILENAME}\`. If a \`${DRAFT_FILENAME}\` \
already exists, first read_file it, then improve it and write it back.

A great README for this crate has THREE qualities, each scored 0–10 by a reviewer:
  1. CLARITY: a crisp one-line summary, then prose a newcomer can follow.
  2. COMPLETENESS: what the crate does, how to add it to Cargo.toml, the main API \
surface, and an error/edge-cases note.
  3. EXAMPLE QUALITY: at least one fenced \`\`\`rust code block showing a real call \
and its expected result.

Write the BEST README you can, then report that you are done.`;

/**
 * System prompt shared by the build agent. (The judge model is prompted
 * separately by the evaluator; it does not share this prompt.) Reinforces the
 * minimal file-tool contract.
 */
const SYSTEM_PROMPT = `You write developer documentation in Markdown. Your only tools are write_file \
(save a file to the workspace) and read_file (read a file back). You have no \
shell and cannot run code. When asked to write or improve the README: read any \
existing file first, write the improved Markdown with write_file, then say you \
are done.`;

/**
 * The rubric handed to the judge model. Kept separate from the build prompt so
 * the judge scores independently of how the writer was instructed.
 */
const JUDGE_RUBRIC = `You are a strict technical-documentation reviewer. Score the README below on \
THREE dimensions, each an integer from 0 to 10:
  - CLARITY: is there a crisp one-line summary and prose a newcomer can follow?
  - COMPLETENESS: does it cover what the crate does, how to add it to Cargo.toml, \
the main API, and an error/edge-cases note?
  - EXAMPLE_QUALITY: is there at least one fenced \`\`\`rust block with a real call \
and expected result?

Reply with EXACTLY these three lines and nothing else:
clarity: <0-10>
completeness: <0-10>
example_quality: <0-10>`;

/**
 * The propose-phase output contract (`propose-schema`). Post-#119,
 * `HillClimbing`'s `inner` (propose) slot is STRUCTURED: a bare `ReAct` there
 * must declare an `output` schema so each iteration yields a scorable candidate
 * ({@link ExecutionRegistry.validate} enforces this via its structured-slot
 * check). The build agent rewrites `DRAFT_FILENAME`; this advertises the path it
 * wrote.
 */
function proposeSchema(): unknown {
  return {
    type: "object",
    properties: {
      file: {
        type: "string",
        description: "Path the candidate draft was written to.",
      },
      summary: { type: "string", description: "What this iteration changed." },
    },
    required: ["file"],
  };
}

/**
 * The {@link ExecutionRegistry} the composed strategy's handles resolve against.
 * Only `propose-schema` is EXPLICIT; the builder default-fills the empty agent /
 * toolset handles (`reactPerLoop`) AND the empty-key metric evaluator from
 * `.metricEvaluator(..)` at `build`. So the HillClimbing `evaluator` stays the
 * EMPTY handle (`""`).
 */
export function buildRegistry(): ExecutionRegistry {
  return ExecutionRegistry.builder()
    .schema("propose-schema", proposeSchema())
    .build();
}

/**
 * The post-#119 composed strategy: `HillClimbing(inner: ReAct, evaluator)`. The
 * propose leaf carries the `propose-schema` output contract (required for the
 * structured `propose` slot) and a `per_loop(perIterBudget)` build budget. The
 * `evaluator` is the EMPTY handle (`""`), which the builder default-fills from
 * `.metricEvaluator(..)`. `maxStagnation` / `minImprovementDelta` are now BARE
 * values (no longer `Option`-wrapped), and the direction enum is
 * `HillClimbingDirection` (renamed from `OptimizationDirection`). Old flat shape
 * was `{ kind: "hill_climbing", direction, max_stagnation, ... }`.
 */
export function hillClimbingStrategy(perIterBudget: number): LoopStrategy {
  return {
    kind: "hill_climbing",
    inner: { ...reactPerLoop(perIterBudget), output: "propose-schema" },
    direction: "maximize",
    max_stagnation: MAX_STAGNATION,
    revert_on_no_improvement: true,
    min_improvement_delta: 0,
    evaluator: "",
  };
}

// ============================================================================
// ReadmeQualityEvaluator — the example-local `MetricEvaluator`.
// ============================================================================

/**
 * Scores `workspace/README.md` by reading it through the {@link SandboxProvider}
 * then making a SEPARATE judge-model call that returns three sub-scores. The
 * value reported to the harness is `total / TOTAL_MAX`, normalized to `[0,1]`,
 * with `direction = "maximize"`.
 *
 * SPEC NOTE: this replaces the shipped `LlmJudgeEvaluator`, which scores a
 * fixed construction-time string and so cannot observe the evolving draft. It
 * returns a {@link metric.MetricOutcome} tagged value — never throws — so a
 * malformed judge reply reads as a poor score, a normal loop outcome.
 */
class ReadmeQualityEvaluator implements metric.MetricEvaluator {
  constructor(private readonly judge: ModelInterface) {}

  /**
   * Parse a `name: <int>` line, clamped to `[0, DIMENSION_MAX]`. A missing or
   * unparseable line scores 0 — a malformed judge reply must not crash the run.
   */
  private static parseDimension(text: string, name: string): number {
    const prefix = `${name}:`;
    for (const line of text.split("\n")) {
      const lower = line.trim().toLowerCase();
      if (lower.startsWith(prefix)) {
        const rest = lower.slice(prefix.length).trim().split(/\s+/)[0];
        const n = Number.parseInt(rest ?? "", 10);
        if (Number.isInteger(n)) return Math.max(0, Math.min(DIMENSION_MAX, n));
      }
    }
    return 0;
  }

  async evaluate(
    sandbox: SandboxProvider,
    _sessionState: termination.SessionStateSnapshot,
    signal?: AbortSignal,
  ): Promise<metric.MetricOutcome> {
    const start = Date.now();

    // Read the current draft through the sandbox root, exactly as the core
    // evaluators do. A missing draft (e.g. the baseline before the agent has
    // written anything) scores 0 rather than erroring.
    const root = sandbox.workspaceRoot?.() ?? "";
    const draftPath = join(root, DRAFT_FILENAME);
    const draft = existsSync(draftPath) ? readFileSync(draftPath, "utf8") : "";

    let clarity = 0;
    let completeness = 0;
    let example = 0;

    if (draft.trim().length === 0) {
      console.log(
        `\n── evaluator: no draft on disk yet (baseline) — total 0/${TOTAL_MAX} ──`,
      );
    } else {
      const request: ModelRequest = {
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `${JUDGE_RUBRIC}\n\n----- README under review -----\n${draft}`,
            },
          },
        ],
        tools: [],
        params: { stop_sequences: [] },
        stream: false,
      };

      let text: string;
      try {
        const response = await this.judge.call(request, signal);
        text = response.content
          .map((b) =>
            b.type === "text" || b.type === "thinking" ? b.text : "",
          )
          .join("\n");
      } catch (err) {
        return {
          kind: "err",
          error: {
            kind: "execution_failed",
            reason: `judge model call failed: ${err instanceof Error ? err.message : String(err)}`,
          },
        };
      }

      clarity = ReadmeQualityEvaluator.parseDimension(text, "clarity");
      completeness = ReadmeQualityEvaluator.parseDimension(
        text,
        "completeness",
      );
      example = ReadmeQualityEvaluator.parseDimension(text, "example_quality");

      console.log(`\n── evaluator: scored draft (${draft.length} bytes) ──`);
      console.log(draft);
      console.log(`  clarity        : ${clarity}/${DIMENSION_MAX}`);
      console.log(`  completeness   : ${completeness}/${DIMENSION_MAX}`);
      console.log(`  example quality: ${example}/${DIMENSION_MAX}`);
    }

    const total = clarity + completeness + example;

    // SPEC NOTE: the threshold is DISPLAY-ONLY. We annotate the line; we do NOT
    // halt the loop here. The harness halts on stagnation/budget.
    const crossed =
      total >= SCORE_THRESHOLD ? "  ★ crossed target threshold" : "";
    console.log(`  TOTAL          : ${total}/${TOTAL_MAX}${crossed}`);

    const value = total / TOTAL_MAX;
    return {
      kind: "ok",
      result: {
        value,
        raw_output: draft,
        duration: (Date.now() - start) / 1000,
        metadata: {
          clarity: String(clarity),
          completeness: String(completeness),
          example_quality: String(example),
          total: String(total),
        },
      },
    };
  }

  direction(): HillClimbingDirection {
    return "maximize";
  }

  description(): string {
    return `ironwood README quality (clarity+completeness+example, /${TOTAL_MAX})`;
  }
}

// ============================================================================
// ReportingObservability — prints each `hill_climbing_iteration` decision.
// ============================================================================

/**
 * An {@link observability.ObservabilityProvider} that extends the in-memory
 * reference provider (so every other span buffers exactly as before) but
 * additionally PRINTS each `hill_climbing_iteration` warn event. This is the
 * seam the harness uses to report the per-iteration keep/revert decision — the
 * evaluator prints the scores, this prints what the loop DID with them. The
 * harness emits exactly one such event per iteration (baseline included).
 */
class ReportingObservability extends observability.InMemoryObservabilityProvider {
  constructor(private readonly maxIterations: number) {
    super();
  }

  override emitWarn(span: observability.WarnSpan): void {
    const event = span.event;
    if (event.warn === "hill_climbing_iteration") {
      // `iteration` is 0-based on the wire (0 = baseline). Display 1-based.
      const n = event.iteration + 1;
      const value =
        event.metric_value == null ? "n/a" : event.metric_value.toFixed(3);
      const delta =
        event.delta == null
          ? "—"
          : `${event.delta >= 0 ? "+" : ""}${event.delta.toFixed(3)}`;
      const revertedNote = event.reverted
        ? " (workspace git-reset to best-so-far)"
        : "";
      console.log(
        `\n══ iteration ${n}/${this.maxIterations} — ${event.status} ══  metric=${value} (Δ ${delta})${revertedNote}`,
      );
    }
    super.emitWarn(span);
  }
}

// ============================================================================
// main
// ============================================================================

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  const modelId =
    argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "llama3.2";
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;

  // Iteration budget: CLI flag wins, then env var, then MAX_ITERATIONS.
  const maxIterations = parsePositiveInt(
    argValue(args, "--max-iterations") ?? process.env.SPORE_MAX_ITERATIONS,
    MAX_ITERATIONS,
  );

  const prompt = argValue(args, "--prompt") ?? TASK_PROMPT;

  // The agent edits this example's `workspace/` in place. Resolve it relative to
  // this source file so `pnpm start` works from anywhere, and create it if
  // missing — the sandbox requires an existing root, which it canonicalizes at
  // construction.
  const here = dirname(fileURLToPath(import.meta.url));
  const workspaceRoot = join(here, "..", "workspace");
  mkdirSync(workspaceRoot, { recursive: true });
  const draftPath = join(workspaceRoot, DRAFT_FILENAME);

  // git-init the workspace so `revert_on_no_improvement`'s `git reset --hard`
  // has a clean baseline to return to. Idempotent: skip if already a repo.
  initGitWorkspace(workspaceRoot);

  // Two model instances on the same Ollama endpoint: one drives the build agent
  // (writing the README), one is the judge the evaluator calls to score it.
  const buildModel = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const judgeModel = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const evaluator = new ReadmeQualityEvaluator(judgeModel);
  const obs = new ReportingObservability(maxIterations);

  // Build harness: conversational preset, workspace sandbox, the minimal file
  // tool set (write_file + read_file), shared system prompt, the observability
  // sink, the registry (carrying the `propose-schema` output contract), and the
  // metric evaluator (folded into the default-key evaluator handle via the fluent
  // `.metricEvaluator(..)` setter — #124).
  const sandbox = new WorkspaceScopedSandbox({ root: workspaceRoot });
  const harness = HarnessBuilder.conversational(buildModel)
    .sandbox(sandbox)
    .registry(buildRegistry())
    .tool(StandardTools.writeFile())
    .tool(StandardTools.readFile())
    .systemPrompt(SYSTEM_PROMPT)
    .observability(obs)
    .metricEvaluator(evaluator)
    .build();

  // THE STRATEGY. No loop code below — the harness runs the climb. The propose
  // leaf carries the `propose-schema` output contract (required for the structured
  // `propose` slot) and a `per_loop(PER_ITER_BUDGET)` build budget; `max_turns`
  // bounds the NUMBER OF ITERATIONS (the budget ceiling), and `max_stagnation` can
  // halt sooner. The empty `evaluator` handle resolves to the metric evaluator
  // above. SPEC NOTE: there is no score-threshold field — by design.
  const task = newTask(
    prompt,
    SessionId.generate(),
    hillClimbingStrategy(PER_ITER_BUDGET),
    { max_turns: maxIterations },
  );

  console.log(`model         : ${modelId}`);
  console.log(`base url      : ${baseUrl}`);
  console.log(`workspace     : ${workspaceRoot}`);
  console.log("strategy      : HillClimbing (score → keep/revert → climb)");
  console.log("direction     : maximize (higher README score is better)");
  console.log(
    `max iterations: ${maxIterations} (budget ceiling — NOT a success target)`,
  );
  console.log(
    `max stagnation: ${MAX_STAGNATION} (halt after this many non-improvements)`,
  );
  console.log(
    `threshold     : ${SCORE_THRESHOLD}/${TOTAL_MAX} — DISPLAY ONLY (★ marks it; never halts)`,
  );
  console.log(
    `\nThe agent will draft and refine \`${DRAFT_FILENAME}\`; each iteration a judge model`,
  );
  console.log(
    "scores it on three dimensions, and the loop keeps the best — reverting the rest —",
  );
  console.log(
    "until it stops improving (stagnation) or the budget is spent. There is no PASS.\n",
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

  if (
    result.kind === "failure" &&
    result.reason.kind === "stagnation_limit_reached"
  ) {
    reportBest(result.reason.best_metric, draftPath);
    console.log(
      `\n■ HALTED ON STAGNATION — ${result.reason.iterations} consecutive non-improving iteration(s).`,
    );
    console.log(
      "This is the NORMAL terminal outcome for HillClimbing: it stopped because it",
    );
    console.log(
      "could not improve, not because it hit a target. The file on disk is best-so-far.",
    );
    return;
  }

  if (result.kind === "failure" && result.reason.kind === "budget_exceeded") {
    reportBest(null, draftPath);
    console.log(
      `\n■ HALTED ON BUDGET — exhausted the iteration ceiling (${result.reason.limit_type}).`,
    );
    console.log(
      "Also a normal terminal outcome: the climb ran out of budget while still",
    );
    console.log(
      "(possibly) improving. The file on disk is the best-so-far draft.",
    );
    return;
  }

  if (
    result.kind === "failure" &&
    result.reason.kind === "hill_climbing_misconfigured"
  ) {
    console.error(`\nHillClimbing misconfigured: ${result.reason.reason}`);
    process.exit(1);
  }

  if (result.kind === "success") {
    // HillClimbing does not normally return success (it has no success
    // condition); surface it honestly if a future core revision does.
    reportBest(null, draftPath);
    console.log(
      `\n■ run returned success after ${result.turns} turn(s) — best-so-far draft on disk.`,
    );
    return;
  }

  console.error(
    `\nrun did not complete as expected: ${JSON.stringify(result)}`,
  );
  process.exit(1);
}

/** Print the best-so-far metric (when known) and the final draft on disk. */
function reportBest(bestMetric: number | null, draftPath: string): void {
  if (bestMetric != null && Number.isFinite(bestMetric)) {
    const total = Math.round(bestMetric * TOTAL_MAX);
    console.log(
      `\n── best score seen: ${total}/${TOTAL_MAX} (normalized ${bestMetric.toFixed(3)}) ──`,
    );
  }
  if (existsSync(draftPath)) {
    console.log(
      `\n── final draft (${draftPath}) ──\n${readFileSync(draftPath, "utf8")}`,
    );
  } else {
    console.log(`\n(no draft was written to ${draftPath})`);
  }
}

/**
 * `git init` the workspace and make an initial commit if it is not already a
 * repo, so `revert_on_no_improvement`'s `git reset --hard` has a baseline.
 * Best-effort and idempotent: a missing `git` or an existing repo is fine.
 */
function initGitWorkspace(root: string): void {
  if (existsSync(join(root, ".git"))) return;
  const git = (gitArgs: string[]): boolean =>
    spawnSync("git", gitArgs, { cwd: root, stdio: "ignore" }).status === 0;
  if (!git(["init"])) return;
  // Local identity so the initial commit succeeds without global git config.
  git(["config", "user.email", "example@spore-core.invalid"]);
  git(["config", "user.name", "spore-core example"]);
  git(["add", "-A"]);
  // An empty initial commit is fine if the dir is otherwise empty.
  git(["commit", "--allow-empty", "-m", "baseline"]);
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
 *  `hillClimbingStrategy`). */
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
