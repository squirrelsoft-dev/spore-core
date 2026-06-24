# Example 10 — `HillClimbing`: scored iterative refinement

> **The quality loop is a harness concern, not application logic.** You wire a
> strategy, a scoring oracle, and an observability sink. The harness runs the
> climb. You write no loop code.

This example demonstrates the [`HillClimbing`] loop strategy: the agent edits
**one file in place** (`workspace/README.md`) across iterations; after every
iteration a scoring oracle grades the draft; the harness **keeps** a draft that
strictly improves the score and **reverts** one that does not; the loop halts
when it **stops improving** (stagnation) or **runs out of budget**.

The agent's task here is to write the README for a fictional Rust crate
`ironwood` (a semver parser). A judge model scores each draft on three
dimensions — **Clarity**, **Completeness**, **Example quality** — 0–10 each, for
a total out of 30, normalized to `[0,1]` for the strategy.

## The teaching point: no PASS, only best-so-far

This is the sharp contrast with **[example 09 — `SelfVerifying`](../09-self-verifying/)**:

| | 09 SelfVerifying | 10 HillClimbing |
| --- | --- | --- |
| Exit condition | **Binary** — a `Verifier` returns PASS | **None** — there is no target to hit |
| Terminal outcome | `Success` (PASS) or exhausted (`SelfVerifyExhausted`) | `StagnationLimitReached` or `BudgetExceeded` |
| What "done" means | "the work is correct" | "the work stopped improving" |
| Per-iteration decision | revise on FAIL | keep if better, else revert |
| Result on disk | the passing draft | the **best-so-far** draft |

SelfVerifying asks *"is it right yet?"* and can answer yes. HillClimbing only
ever asks *"is this better than the best I've seen?"* — it never declares
victory, it just notices it has stopped climbing. The honest framing is:
**there is no PASS. There is only best-so-far.**

### `SCORE_THRESHOLD` is a display marker, not a termination condition

The original task framing for this example was *"climb until total ≥ 25/30"*.
That does not match the real strategy (see "Spec note" below). We keep
`SCORE_THRESHOLD = 25` purely as a **display annotation**: when a draft's total
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
3. **Score.** [`ReadmeQualityEvaluator`] (in `src/main.rs`) reads `README.md`
   through the `SandboxProvider`, prompts a *separate judge model* with the
   rubric, parses the three sub-scores, and returns `total / 30` as the metric
   (`direction = Maximize`).
4. **Keep or revert.** The harness applies [`should_keep`]: a *strictly* better
   score is **kept**; an equal-or-worse score is **discarded**, and because
   `revert_on_no_improvement = true` the workspace is `git reset --hard`-ed back
   to the best-so-far draft.
5. **Halt.** After `MAX_STAGNATION` (2) consecutive non-improvements the loop
   halts with `StagnationLimitReached`; if it keeps improving it eventually halts
   on the `max_turns` budget (`MAX_ITERATIONS`, default 6) with `BudgetExceeded`.

### The two output seams

The per-iteration output is split across the two seams the harness actually
exposes, so the printing matches the real architecture rather than faking a
single log:

- **The evaluator** prints the draft and the rubric breakdown (the three
  sub-scores + total + the `★` annotation) — it is the only component that sees
  the rubric.
- **A custom `ObservabilityProvider`** ([`ReportingObservability`]) handles
  [`WarnEvent::HillClimbingIteration`] and prints what the loop *did*:
  `kept` / `discarded`, the metric value, the delta, and whether the workspace
  was reverted. The harness emits exactly one such event per iteration.

## How it's assembled

The harness is built from the **`HarnessBuilder::hill_climber`** preset (SC-8),
which folds in the two things every climb needs: the metric evaluator (the
`HillClimbing` strategy requires one) and `EscalationMode::AutoContinue` — so a
build agent that spends its per-iteration step budget keeps working in-process
instead of pausing the climb. The preset deliberately leaves the
workspace-specific bits to the caller (climbs vary), so this example adds the
read-write sandbox, the `write_file` + `read_file` tools, the build system prompt,
the propose-schema registry, and the observability sink:

```rust
let harness = HarnessBuilder::hill_climber(build_model, evaluator) // evaluator + AutoContinue
    .sandbox(Arc::new(sandbox))                                    // read-WRITE workspace
    .tool(StandardTools::write_file())
    .tool(StandardTools::read_file())
    .system_prompt(SYSTEM_PROMPT)
    .registry(build_registry())                                    // propose-schema
    .observability(observability)                                  // prints keep/revert
    .build();
```

## Constants (top of `src/main.rs`)

| Constant | Value | Meaning |
| --- | --- | --- |
| `MAX_ITERATIONS` | `6` | Iteration **budget** → `BudgetLimits.max_turns`. Not a target. |
| `MAX_STAGNATION` | `2` | Consecutive non-improvements before halt → `max_stagnation`. |
| `SCORE_THRESHOLD` | `25` | **Display only.** Marks a draft `★ crossed target threshold`. Never halts. |
| `DIMENSION_MAX` | `10` | Max score per rubric dimension. |
| `TOTAL_MAX` | `30` | Max total (`3 × DIMENSION_MAX`). |

## The `git`-init'd workspace

`revert_on_no_improvement` reverts via the sandbox's VCS, i.e. `git reset
--hard`. So the example **`git init`s** `workspace/` and makes an initial commit
at startup (idempotent — skipped if it is already a repo). Without a git
baseline a revert would have nothing to reset to.

## Run it

```sh
ollama serve &
ollama pull qwen2.5-coder:7b

# A TOOL-CAPABLE model is required — the build agent must call write_file. A
# narrate-only small model (e.g. llama3.2 3B) never acts, so the draft stays empty
# and every iteration scores 0. Pass one with --model:
cargo run -- --model qwen2.5-coder:7b
cargo run -- --model qwen2.5-coder:7b --max-iterations 8   # widen the budget
```

> The code default is still `llama3.2` (so `cargo run` with no flags starts), but
> for a real climb pass a tool-capable model as above.

Configuration mirrors example 09: `--model` / `SPORE_OLLAMA_MODEL`,
`SPORE_OLLAMA_BASE_URL`, and `--max-iterations` / `SPORE_MAX_ITERATIONS`. See
`.env.example`.

> This example needs a running model (Ollama by default). It is **not** part of
> the build/clippy/fmt gate, which is all that CI checks — the same bar as 09.

## Rough edges (the honest part)

HillClimbing against a small local model is genuinely noisy, and it is worth
seeing *why* rather than pretending the demo is clean:

- **Judge variance reads as signal.** The judge re-scores a freeform README
  every iteration. A small model's three sub-scores can wobble by a point or two
  for an *unchanged* draft. The keep/revert logic then interprets that wobble as
  "improvement" or "regression" — so you will see drafts kept or reverted on
  noise, not real edits. A larger / hosted model scores more steadily.
- **Equal is discarded.** [`should_keep`] requires a *strictly* better score
  (`min_improvement_delta` is `None` ⇒ delta must exceed `0.0`). A draft that
  re-scores identically is reverted and counts toward stagnation. A genuinely
  good draft can therefore be abandoned simply because the judge could not push
  the number higher.
- **Stagnation can halt early.** With `MAX_STAGNATION = 2`, two unlucky judge
  calls in a row end the run — even mid-improvement. That is the strategy
  working as designed (it cannot tell "stuck" from "unlucky"), not a bug.
- **No PASS to chase.** Because there is no success condition, a run that climbs
  to `28/30` and a run that plateaus at `12/30` both terminate the same way —
  on stagnation or budget. The *score* is the outcome; the *halt reason* is just
  how it stopped.

These are the real properties of optimization-style loops. The point of the
example is to make them visible, not to hide them behind a tidy PASS.

## Where to look

- `src/main.rs` — the whole example. Start at `main`, then read
  `ReadmeQualityEvaluator` (the scoring oracle) and `ReportingObservability`
  (the per-iteration decision printer).
- `rust/crates/spore-core/src/harness.rs` — `run_hill_climbing` and the
  `LoopStrategy::HillClimbing` config fields.
- `rust/crates/spore-core/src/metric.rs` — the `MetricEvaluator` trait,
  `MetricResult`, and `should_keep`.

[`HillClimbing`]: ../../../rust/crates/spore-core/src/harness.rs
[`ReadmeQualityEvaluator`]: src/main.rs
[`ReportingObservability`]: src/main.rs
[`WarnEvent::HillClimbingIteration`]: ../../../rust/crates/spore-core/src/observability.rs
[`should_keep`]: ../../../rust/crates/spore-core/src/metric.rs
