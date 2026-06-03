# 10 — Hill-climbing (a scored optimization loop as a harness strategy)

> **The quality loop is a harness concern, not application logic.** You wire a
> strategy, a scoring oracle, and an observability sink. The harness runs the
> climb. You write no loop code.

This example wires the **`HillClimbing`** loop strategy: the agent edits **one
file in place** (`workspace/README.md`) across iterations; after every iteration
a scoring oracle grades the draft; the harness **keeps** a draft that strictly
improves the score and **reverts** one that does not; the loop halts when it
**stops improving** (stagnation) or **runs out of budget**.

The agent's task here is to write the README for a fictional Rust crate
`ironwood` (a semver parser). A judge model scores each draft on three
dimensions — **Clarity**, **Completeness**, **Example quality** — 0–10 each, for
a total out of 30, normalized to `[0,1]` for the strategy.

## The teaching point: no PASS, only best-so-far

This is the sharp contrast with **[example 09 — `SelfVerifying`](../09-self-verifying/)**:

|                        | 09 SelfVerifying                          | 10 HillClimbing                                  |
| ---------------------- | ----------------------------------------- | ------------------------------------------------ |
| Exit condition         | **Binary** — a `Verifier` returns PASS    | **None** — there is no target to hit             |
| Terminal outcome       | `success` (PASS) or `self_verify_exhausted` | `stagnation_limit_reached` or `budget_exceeded`  |
| What "done" means      | "the work is correct"                     | "the work stopped improving"                     |
| Per-iteration decision | revise on FAIL                            | keep if better, else revert                      |
| Result on disk         | the passing draft                         | the **best-so-far** draft                        |

SelfVerifying asks _"is it right yet?"_ and can answer yes. HillClimbing only
ever asks _"is this better than the best I've seen?"_ — it never declares
victory, it just notices it has stopped climbing. The honest framing is:
**there is no PASS. There is only best-so-far.**

### `SCORE_THRESHOLD` is a display marker, not a termination condition

The original task framing for this example was _"climb until total ≥ 25/30"_.
That does not match the real strategy (see "Spec note" below). We keep
`SCORE_THRESHOLD = 25` purely as a **display annotation**: when a draft's total
crosses it, the evaluator prints `★ crossed target threshold` on that line. The
loop does **not** stop there. It keeps climbing until stagnation or budget,
exactly as the strategy is built to. If you want to "stop at a score," that is a
_different_ strategy (`SelfVerifying` with a threshold verifier), not this one.

## What it demonstrates

- `LoopStrategy.hill_climbing` — the score → keep/revert → climb cycle, run by the harness.
- A custom `MetricEvaluator` as the scoring oracle, reading the evolving draft each iteration.
- The keep/revert decision made **visible** through the observability seam.
- A real-world stress test of the strategy against a live model — including the
  rough edges (documented honestly below).

## How it works

1. **Baseline (iteration 0).** The evaluator scores the workspace before any
   agent turn. With an empty workspace that is `0/30` — the floor to climb from.
2. **Each iteration.** The build agent reads the current `README.md`, improves
   it, and writes it back via `write_file`. The harness then calls the metric
   evaluator.
3. **Score.** `ReadmeQualityEvaluator` (in `src/main.ts`) reads `README.md`
   through the `SandboxProvider`, prompts a _separate judge model_ with the
   rubric, parses the three sub-scores, and returns `total / 30` as the metric
   (`direction = "maximize"`).
4. **Keep or revert.** The harness applies `shouldKeep`: a _strictly_ better
   score is **kept**; an equal-or-worse score is **discarded**, and because
   `revert_on_no_improvement = true` the workspace is `git reset --hard`-ed back
   to the best-so-far draft.
5. **Halt.** After `MAX_STAGNATION` (2) consecutive non-improvements the loop
   halts with `stagnation_limit_reached`; if it keeps improving it eventually
   halts on the `max_turns` budget (`MAX_ITERATIONS`, default 6) with
   `budget_exceeded`.

### The two output seams

The per-iteration output is split across the two seams the harness actually
exposes, so the printing matches the real architecture rather than faking a
single log:

- **The evaluator** prints the draft and the rubric breakdown (the three
  sub-scores + total + the `★` annotation) — it is the only component that sees
  the rubric.
- **A custom `ObservabilityProvider`** (`ReportingObservability`) handles the
  `hill_climbing_iteration` warn event and prints what the loop _did_:
  `kept` / `discarded`, the metric value, the delta, and whether the workspace
  was reverted. The harness emits exactly one such event per iteration.

### The wiring (so the other language ports can mirror it)

```ts
const evaluator = new ReadmeQualityEvaluator(judgeModel); // the scoring oracle
const obs = new ReportingObservability(maxIterations); // the per-iteration printer

// The TS HarnessBuilder has fluent setters for most seams — but NOT for the
// metric evaluator (the Rust builder has `.metric_evaluator(...)`; the TS one
// has not grown it yet). So assemble the config, attach the evaluator, and
// construct the harness directly — exactly how the core hill-climbing tests
// wire it. The observability sink DOES have a fluent setter.
const config = HarnessBuilder.conversational(buildModel)
  .sandbox(sandbox)
  .tool(StandardTools.writeFile()) // builder writes the draft
  .tool(StandardTools.readFile()) // evaluator reads it back
  .systemPrompt(SYSTEM_PROMPT)
  .observability(obs)
  .buildConfig();
config.metricEvaluator = evaluator;
const harness = new StandardHarness(config);

const task = newTask(
  prompt,
  SessionId.generate(),
  {
    kind: "hill_climbing",
    direction: "maximize",
    max_stagnation: MAX_STAGNATION,
    revert_on_no_improvement: true,
    min_improvement_delta: null,
  },
  { max_turns: maxIterations },
);

await harness.run({ task }); // no loop code here
```

## Constants (top of `src/main.ts`)

| Constant          | Value | Meaning                                                                      |
| ----------------- | ----- | ---------------------------------------------------------------------------- |
| `MAX_ITERATIONS`  | `6`   | Iteration **budget** → `BudgetLimits.max_turns`. Not a target.               |
| `MAX_STAGNATION`  | `2`   | Consecutive non-improvements before halt → `max_stagnation`.                 |
| `SCORE_THRESHOLD` | `25`  | **Display only.** Marks a draft `★ crossed target threshold`. Never halts.   |
| `DIMENSION_MAX`   | `10`  | Max score per rubric dimension.                                              |
| `TOTAL_MAX`       | `30`  | Max total (`3 × DIMENSION_MAX`).                                             |

## The `git`-init'd workspace

`revert_on_no_improvement` reverts via the sandbox's VCS, i.e. `git reset
--hard`. So the example **`git init`s** `workspace/` and makes an initial commit
at startup (idempotent — skipped if it is already a repo). Without a git
baseline a revert would have nothing to reset to. The example's `.gitignore`
keeps the generated `workspace/README.md` and the nested `workspace/.git` out of
the repo — they are local run state.

## Spec note — why this diverges from issue #99's original framing (Option A)

The original issue asked the agent to "climb until total ≥ 25/30 or max
iterations". Planning (#99 spec-resolution comment) established that framing does
**not** match the real `HillClimbing` strategy in spore-core:

- There is no score-threshold success condition. The loop keeps/reverts on
  _relative_ improvement and halts on stagnation/budget — it never compares the
  metric against an absolute target.
- `MAX_ITERATIONS` is not a HillClimbing parameter; iterations are bounded by
  `BudgetLimits.max_turns`. The `MAX_ITERATIONS` constant maps there.
- The shipped `LlmJudgeEvaluator` scores a fixed construction-time string, so it
  cannot see the evolving draft. This example therefore ships a small
  example-local `ReadmeQualityEvaluator` that reads `workspace/README.md`
  through the sandbox each iteration before scoring.

The resolution (**Option A**, no core change) keeps `SCORE_THRESHOLD` as a
display annotation only and frames the contrast with 09 honestly: SelfVerifying
has a binary PASS; HillClimbing has no PASS — only best-so-far.

## Rough edges (the honest part)

HillClimbing against a small local model is genuinely noisy, and it is worth
seeing _why_ rather than pretending the demo is clean:

- **Judge variance reads as signal.** The judge re-scores a freeform README
  every iteration. A small model's three sub-scores can wobble by a point or two
  for an _unchanged_ draft. The keep/revert logic then interprets that wobble as
  "improvement" or "regression" — so you will see drafts kept or reverted on
  noise, not real edits. A larger / hosted model scores more steadily.
- **Equal is discarded.** `shouldKeep` requires a _strictly_ better score
  (`min_improvement_delta` is `null` ⇒ delta must exceed `0.0`). A draft that
  re-scores identically is reverted and counts toward stagnation. A genuinely
  good draft can therefore be abandoned simply because the judge could not push
  the number higher.
- **Stagnation can halt early.** With `MAX_STAGNATION = 2`, two unlucky judge
  calls in a row end the run — even mid-improvement. That is the strategy working
  as designed (it cannot tell "stuck" from "unlucky"), not a bug.
- **No PASS to chase.** Because there is no success condition, a run that climbs
  to `28/30` and a run that plateaus at `12/30` both terminate the same way — on
  stagnation or budget. The _score_ is the outcome; the _halt reason_ is just how
  it stopped.
- **Prose-instead-of-tool-call — mostly handled.** If the build agent describes
  the write in prose instead of calling `write_file`, the `conversational`
  preset's **adaptive prompt-based tool-calling repair** (in all four cores)
  nudges the model once and escalates to prompt-based `<tool_call>` markers that
  are parsed back into native tool calls — so the write usually still lands. The
  honest caveat: it is a one-shot escalation; a model that keeps answering in
  prose still won't write the file, and then the draft never improves.

These are the real properties of optimization-style loops. The point of the
example is to make them visible, not to hide them behind a tidy PASS.

The maintainer-verified known-good model is **`gemma4:31b-cloud`** (an Ollama
_cloud_ model): it gives a steadier demo and, because it's served through Ollama,
needs **no code edit** — just set `SPORE_OLLAMA_MODEL=gemma4:31b-cloud`. As a
secondary alternative, spore-core ships an `AnthropicModelInterface`; to target
it, swap the two `OllamaModelInterface.withBaseUrl(...)` constructions in
`src/main.ts`. This example wires Ollama by default to keep it key-free and to
honestly exercise the strategy under stress.

## Prerequisites

- [Ollama](https://ollama.com) running locally:
  ```sh
  ollama serve &
  ollama pull llama3.2     # or a stronger model, e.g. qwen2.5-coder:7b
  ```
- `git` on `PATH` (the example `git init`s `workspace/` so reverts work).
- `pnpm install` (from this directory) to link the local `@spore/core` and
  `@spore/tools` workspace packages.

## Run it

```sh
pnpm install
pnpm start                                # llama3.2, 6-iteration budget
pnpm start -- --max-iterations 8          # widen the budget
pnpm start -- --model qwen2.5-coder:7b    # a stronger local model
pnpm start -- --prompt "…custom task…"    # override the task
```

Or via env (`.env.example`): `SPORE_OLLAMA_MODEL`, `SPORE_OLLAMA_BASE_URL`,
`SPORE_MAX_ITERATIONS`.

> This example needs a running model (Ollama by default). It is **not** part of
> the lint/format gate, which is all that CI checks — the same bar as 09.

## CLI flags

| Flag                   | Env                     | Default                  | Meaning                                    |
| ---------------------- | ----------------------- | ------------------------ | ------------------------------------------ |
| `--model <id>`         | `SPORE_OLLAMA_MODEL`    | `llama3.2`               | Ollama model for builder **and** judge     |
| `--max-iterations <n>` | `SPORE_MAX_ITERATIONS`  | `6`                      | Iteration budget (`max_turns`)             |
| `--prompt <text>`      | —                       | the ironwood README task | Override the task the agent solves         |
| (base url)             | `SPORE_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint                            |

## Where to look

- `src/main.ts` — the whole example. Start at `main`, then read
  `ReadmeQualityEvaluator` (the scoring oracle) and `ReportingObservability`
  (the per-iteration decision printer).
- `typescript/packages/core/src/harness/standard.ts` — `runHillClimbing` and the
  `hill_climbing` strategy config fields.
- `typescript/packages/core/src/metric/types.ts` — the `MetricEvaluator`
  interface, `MetricResult`/`MetricOutcome`, and `shouldKeep`.
