# 10 — Hill-climbing (scored iterative refinement as a harness strategy)

The first Go example to wire the **`HillClimbing`** loop strategy. The agent
edits **one file in place** (`workspace/README.md`) across iterations; after
every iteration a scoring oracle grades the draft; the harness **keeps** a draft
that strictly improves the score and **reverts** one that does not; the loop
halts when it **stops improving** (stagnation) or **runs out of budget**.

The thesis: **iterative refinement under a scoring oracle is a harness concern,
not application logic.** You do not write a `for score < target { revise() }`
loop. You wire a strategy, a `MetricEvaluator`, and an observability sink — and
the harness runs the climb for you. The application code is one `Task` with
`LoopStrategy{Kind: StrategyHillClimbing}`, a metric evaluator, and a git-init'd
workspace.

The agent's task here is to write the README for a fictional Rust crate
`ironwood` (a semver parser). A judge model scores each draft on three
dimensions — **Clarity**, **Completeness**, **Example quality** — 0–10 each, for
a total out of 30, normalized to `[0,1]` for the strategy.

## What it demonstrates

- `LoopStrategy{Kind: StrategyHillClimbing}` — the score → keep/revert → climb
  cycle, run by the harness.
- A custom example-local `sporecore.MetricEvaluator` that reads the evolving
  draft each iteration and scores it with a separate judge model.
- The per-iteration keep/revert decision made **visible** through the
  observability `HillClimbingIteration` warn event.
- A real-world stress test of the strategy against a live model — including the
  rough edges (documented honestly below).

## The teaching point: no PASS, only best-so-far

This is the sharp contrast with **[example 09 — `SelfVerifying`](../09-self-verifying/)**:

| | 09 SelfVerifying | 10 HillClimbing |
| --- | --- | --- |
| Exit condition | **Binary** — a `Verifier` returns PASS | **None** — there is no target to hit |
| Terminal outcome | `RunSuccess` (PASS) or exhausted (`HaltSelfVerifyExhausted`) | `HaltStagnationLimitReached` or `HaltBudgetExceeded` |
| What "done" means | "the work is correct" | "the work stopped improving" |
| Per-iteration decision | revise on FAIL | keep if better, else revert |
| Result on disk | the passing draft | the **best-so-far** draft |

SelfVerifying asks *"is it right yet?"* and can answer yes. HillClimbing only
ever asks *"is this better than the best I've seen?"* — it never declares
victory, it just notices it has stopped climbing. The honest framing is:
**there is no PASS. There is only best-so-far.**

### `ScoreThreshold` is a display marker, not a termination condition

The original task framing for this example was *"climb until total ≥ 25/30"*.
That does not match the real strategy (see "Spec note" below). We keep
`ScoreThreshold = 25` purely as a **display annotation**: when a draft's total
crosses it, the evaluator prints `★ crossed target threshold` on that line. The
loop does **not** stop there. It keeps climbing until stagnation or budget,
exactly as the strategy is built to. If you want to "stop at a score," that is a
*different* strategy (`SelfVerifying` with a threshold verifier), not this one.

## How it works

1. **Baseline (iteration 0).** The evaluator scores the workspace before any
   agent turn. With an empty workspace that is `0/30` — the floor to climb from.
2. **Each iteration.** The build agent reads the current `README.md`, improves
   it, and writes it back via `write_file`. The harness then calls the metric
   evaluator.
3. **Score.** `readmeQualityEvaluator` (in `main.go`) reads `README.md` through
   the `SandboxProvider`, prompts a *separate judge model* with the rubric,
   parses the three sub-scores, and returns `total / 30` as the metric
   (`Direction = Maximize` on the strategy payload).
4. **Keep or revert.** The harness applies its keep rule
   (`hillClimbShouldKeep`): a *strictly* better score is **kept**; an
   equal-or-worse score is **discarded**, and because `RevertOnNoImprovement =
   true` the workspace is `git reset --hard`-ed back to the best-so-far draft.
5. **Halt.** After `MaxStagnation` (2) consecutive non-improvements the loop
   halts with `HaltStagnationLimitReached`; if it keeps improving it eventually
   halts on the `MaxTurns` budget (`MaxIterations`, default 6) with
   `HaltBudgetExceeded`.

### The two output seams

The per-iteration output is split across the two seams the harness actually
exposes, so the printing matches the real architecture rather than faking a
single log:

- **The evaluator** prints the draft and the rubric breakdown (the three
  sub-scores + total + the `★` annotation) — it is the only component that sees
  the rubric.
- **A custom `ObservabilityProvider`** (`reportingObservability`) handles the
  `HillClimbingIteration` warn event and prints what the loop *did*: `kept` /
  `discarded`, the metric value, the delta, and whether the workspace was
  reverted. The harness emits exactly one such event per iteration.

### Go wiring asymmetries (worth a look if you are porting)

- The harness-seam `MetricEvaluator` lives in the **root** `sporecore` package
  (its signature is `Evaluate(ctx, sandbox, sessionID, taskID, state)
  (*HillClimbMetricResult, *HillClimbMetricError)` plus `Description()`), not the
  `metric` package. The example implements that consumer-side interface directly.
- The warn event is delivered through the `HarnessObserver` seam. The
  `observability.HarnessBuilder` bridges an `ObservabilityProvider` into that
  seam and **type-asserts it to `WarnEmitter`** before forwarding the warn. So
  `reportingObservability` embeds an `InMemoryObservabilityProvider` (which
  satisfies the full provider interface *and* `WarnEmitter`) and overrides
  `EmitWarn`.
- Mirroring 09, the builder has no `MetricEvaluator` setter, so the example
  builds the config with `observability.ConversationalBuilder(...).BuildConfig()`
  and sets `cfg.MetricEvaluator` on the built config before
  `NewStandardHarness` — the same cfg-field asymmetry 09 uses for `cfg.Verifier`.

## Constants (top of `main.go`)

| Constant | Value | Meaning |
| --- | --- | --- |
| `MaxIterations` | `6` | Iteration **budget** → `BudgetLimits.MaxTurns`. Not a target. |
| `MaxStagnation` | `2` | Consecutive non-improvements before halt → `MaxStagnation`. |
| `ScoreThreshold` | `25` | **Display only.** Marks a draft `★ crossed target threshold`. Never halts. |
| `dimensionMax` | `10` | Max score per rubric dimension. |
| `totalMax` | `30` | Max total (`3 × dimensionMax`). |

## The `git`-init'd workspace

`RevertOnNoImprovement` reverts via the sandbox's VCS, i.e. `git reset --hard`.
So the example **`git init`s** `workspace/` and makes an initial commit at
startup (idempotent — skipped if it is already a repo). Without a git baseline a
revert would have nothing to reset to.

## Run it

```sh
ollama serve &
ollama pull llama3.2

go run .                       # default model llama3.2, 6-iteration budget
go run . --max-iterations 8    # widen the budget
go run . --model qwen2.5-coder:7b
```

Configuration mirrors example 09: `--model` / `SPORE_OLLAMA_MODEL`,
`SPORE_OLLAMA_BASE_URL`, and `--max-iterations` / `SPORE_MAX_ITERATIONS`. See
`.env.example`.

> This example needs a running model (Ollama by default). It is **not** part of
> the build/vet/fmt gate, which is all that CI checks — the same bar as 09.

## Rough edges (the honest part)

HillClimbing against a small local model is genuinely noisy, and it is worth
seeing *why* rather than pretending the demo is clean:

- **Judge variance reads as signal.** The judge re-scores a freeform README
  every iteration. A small model's three sub-scores can wobble by a point or two
  for an *unchanged* draft. The keep/revert logic then interprets that wobble as
  "improvement" or "regression" — so you will see drafts kept or reverted on
  noise, not real edits. A larger / hosted model scores more steadily.
- **Equal is discarded.** The keep rule requires a *strictly* better score
  (`MinImprovementDelta` is `nil` ⇒ delta must exceed `0.0`). A draft that
  re-scores identically is reverted and counts toward stagnation. A genuinely
  good draft can therefore be abandoned simply because the judge could not push
  the number higher.
- **Stagnation can halt early.** With `MaxStagnation = 2`, two unlucky judge
  calls in a row end the run — even mid-improvement. That is the strategy
  working as designed (it cannot tell "stuck" from "unlucky"), not a bug.
- **No PASS to chase.** Because there is no success condition, a run that climbs
  to `28/30` and a run that plateaus at `12/30` both terminate the same way — on
  stagnation or budget. The *score* is the outcome; the *halt reason* is just how
  it stopped.

These are the real properties of optimization-style loops. The point of the
example is to make them visible, not to hide them behind a tidy PASS.

## Where to look

- `main.go` — the whole example. Start at `run`, then read
  `readmeQualityEvaluator` (the scoring oracle) and `reportingObservability`
  (the per-iteration decision printer).
- `../../../go/spore-core/hill_climbing.go` — `runHillClimbing`, the
  `MetricEvaluator` seam, `hillClimbShouldKeep`, and the result/error types.
- `../../../go/spore-core/harness.go` — the `LoopStrategy` HillClimbing fields
  (`Direction`, `MaxStagnation`, `RevertOnNoImprovement`, `MinImprovementDelta`)
  and the `HarnessObserver.EmitHillClimbingIteration` seam.
