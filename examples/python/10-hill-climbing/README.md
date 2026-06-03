# 10 — Hill-climbing (scored iterative refinement as a harness strategy)

The first example to wire the **`HillClimbing`** loop strategy. The agent edits
**one file in place** (`workspace/README.md`) across iterations; after every
iteration a *separate judge model* scores the draft; the harness **keeps** a draft
that strictly improves the score and **reverts** one that does not; the loop halts
when it **stops improving** (stagnation) or **runs out of budget**.

The thesis: **iterative refinement under a scoring oracle is a harness concern,
not application logic.** You do not write a `while score_improving: revise()`
loop. You wire a strategy, a `MetricEvaluator`, and an observability sink — and
the harness runs the score → keep/revert → climb cycle for you. The application
code is one `Task` with `LoopStrategyHillClimbing`, an evaluator, and a sink.

The agent's task here is to write the README for a fictional Rust crate `ironwood`
(a semver parser). A judge model scores each draft on three dimensions —
**Clarity**, **Completeness**, **Example quality** — 0–10 each, for a total out of
30, normalized to `[0,1]` for the strategy.

## What it demonstrates

- `LoopStrategyHillClimbing` — the score/keep/revert/climb cycle, run by the harness.
- A custom `MetricEvaluator` that reads the evolving draft each iteration and
  scores it with a *fresh* judge-model call (the shipped `LlmJudgeEvaluator`
  cannot — see "Spec note" below).
- The per-iteration decision made **visible** through a custom
  `ObservabilityProvider` seam.
- A real-world stress test of the strategy against a live model — including the
  rough edges (documented honestly below).

## The teaching point: no PASS, only best-so-far

This is the sharp contrast with **[example 09 — `SelfVerifying`](../09-self-verifying/)**:

| | 09 SelfVerifying | 10 HillClimbing |
| --- | --- | --- |
| Exit condition | **Binary** — a `Verifier` returns PASS | **None** — there is no target to hit |
| Terminal outcome | `Success` (PASS) or exhausted (`SelfVerifyExhausted`) | `StagnationLimitReached` or `BudgetExceeded` |
| What "done" means | "the work is correct" | "the work stopped improving" |
| Per-iteration decision | revise on FAIL | keep if better, else revert |
| Result on disk | the passing draft | the **best-so-far** draft |

SelfVerifying asks *"is it right yet?"* and can answer yes. HillClimbing only ever
asks *"is this better than the best I've seen?"* — it never declares victory, it
just notices it has stopped climbing. The honest framing is: **there is no PASS.
There is only best-so-far.**

### `SCORE_THRESHOLD` is a display marker, not a termination condition

The original task framing for this example was *"climb until total ≥ 25/30"*. That
does not match the real strategy (see "Spec note" below). We keep `SCORE_THRESHOLD
= 25` purely as a **display annotation**: when a draft's total crosses it, the
evaluator prints `★ crossed target threshold` on that line. The loop does **not**
stop there. It keeps climbing until stagnation or budget, exactly as the strategy
is built to. If you want to "stop at a score," that is a *different* strategy
(`SelfVerifying` with a threshold verifier), not this one.

## How it works

1. **Baseline (iteration 0).** The evaluator scores the workspace before any agent
   turn. With an empty workspace that is `0/30` — the floor to climb from.
2. **Each iteration.** The build agent reads the current `README.md`, improves it,
   and writes it back via `write_file`. The harness then calls the metric evaluator.
3. **Score.** `ReadmeQualityEvaluator` (in `main.py`) reads `README.md` through the
   `SandboxProvider`, prompts a *separate judge model* with the rubric, parses the
   three sub-scores, and returns `total / 30` as the metric (`direction = maximize`).
4. **Keep or revert.** The harness keeps a *strictly* better score; an
   equal-or-worse score is **discarded**, and because `revert_on_no_improvement =
   True` the workspace is `git reset --hard`-ed back to the best-so-far draft.
5. **Halt.** After `MAX_STAGNATION` (2) consecutive non-improvements the loop halts
   with `HaltReasonStagnationLimitReached`; if it keeps improving it eventually
   halts on the `max_turns` budget (`MAX_ITERATIONS`, default 6) with
   `HaltReasonBudgetExceeded`.

You wire three things; the harness does the rest:

```python
evaluator = ReadmeQualityEvaluator(judge_model)   # the scoring oracle
observability = ReportingObservability(max_iterations)  # the per-iteration printer

harness = (
    HarnessBuilder.conversational(build_model)
    .sandbox(sandbox)
    .tool(StandardTools.write_file())   # builder writes the draft
    .tool(StandardTools.read_file())    # builder re-reads it to improve it
    .system_prompt(SYSTEM_PROMPT)
    .metric_evaluator(evaluator)        # REQUIRED for HillClimbing
    .observability(observability)       # the decision seam
    .build()
)

task = Task.new(
    prompt, new_session_id(),
    LoopStrategyHillClimbing(
        direction="maximize",
        max_stagnation=MAX_STAGNATION,
        revert_on_no_improvement=True,
        min_improvement_delta=None,
    ),
    budget=BudgetLimits(max_turns=max_iterations),  # the ITERATION budget
)

await harness.run(HarnessRunOptions(task))   # no loop code here
```

Both `ReadmeQualityEvaluator` and `ReportingObservability` satisfy their
respective Protocols. The evaluator satisfies `MetricEvaluator` **structurally** —
no inheritance, per the Python conventions. `ReportingObservability` *does*
subclass the concrete `InMemoryObservabilityProvider` so it can override only
`emit_warn` and inherit the rest of the (many) trace-recording methods verbatim.

### The two output seams

The per-iteration output is split across the two seams the harness actually
exposes, so the printing matches the real architecture rather than faking a single
log:

- **The evaluator** prints the draft and the rubric breakdown (the three sub-scores
  + total + the `★` annotation) — it is the only component that sees the rubric.
- **A custom `ObservabilityProvider`** (`ReportingObservability`) handles
  `WarnEventHillClimbingIteration` and prints what the loop *did*: `kept` /
  `discarded`, the metric value, the delta, and whether the workspace was reverted.
  The harness emits exactly one such event per iteration (iteration 0 is the
  baseline; its delta is `None`).

## The tool decision — one file seam

The evaluator does not receive the build agent's draft text directly: it must read
the file off disk to score it. So this example wires the **minimal file tool set**
and nothing else:

- `write_file` — the build agent saves (and overwrites) its draft at
  `workspace/README.md`.
- `read_file`  — the build agent reads the current draft back so it can *improve*
  it rather than rewrite from scratch, and the evaluator reads it to score.

No `web_search`, no shell. The loop is the point.

## Constants (top of `main.py`)

| Constant | Value | Meaning |
| --- | --- | --- |
| `MAX_ITERATIONS` | `6` | Iteration **budget** → `BudgetLimits.max_turns`. Not a target. |
| `MAX_STAGNATION` | `2` | Consecutive non-improvements before halt → `max_stagnation`. |
| `SCORE_THRESHOLD` | `25` | **Display only.** Marks a draft `★ crossed target threshold`. Never halts. |
| `DIMENSION_MAX` | `10` | Max score per rubric dimension. |
| `TOTAL_MAX` | `30` | Max total (`3 × DIMENSION_MAX`). |

## Spec note — why this diverges from issue #99's original framing (Option A)

The original issue asked the agent to "climb until total ≥ 25/30 or max
iterations." Planning (#99 spec-resolution comment) established that framing does
**not** match the real `HillClimbing` strategy in spore-core:

- There is no score-threshold success condition; the loop keeps/reverts on
  *relative* improvement and halts on stagnation/budget. It never compares the
  metric against an absolute target.
- `MAX_ITERATIONS` is not a HillClimbing parameter — iterations are bounded by
  `BudgetLimits.max_turns`. The `MAX_ITERATIONS` constant maps there.
- The shipped `LlmJudgeEvaluator` scores a fixed construction-time string, so it
  cannot see the evolving draft. This example ships its own `ReadmeQualityEvaluator`
  that reads `workspace/README.md` through the sandbox each iteration.

Resolution = **Option A** (reframe to the real semantics, no core change), with
`SCORE_THRESHOLD` retained as a display annotation only. The same divergence is
documented in the four-language ports; see `examples/rust/10-hill-climbing` for the
reference implementation.

## The `git`-init'd workspace

`revert_on_no_improvement` reverts via the sandbox's VCS, i.e. `git reset --hard`.
So the example **`git init`s** `workspace/` and makes an initial commit at startup
(idempotent — skipped if it is already a repo). Without a git baseline a revert
would have nothing to reset to.

## Rough edges (honest, because this is also a stress test)

HillClimbing against a small local model is genuinely noisy, and it is worth seeing
*why* rather than pretending the demo is clean:

- **Judge variance reads as signal.** The judge re-scores a freeform README every
  iteration. A small model's three sub-scores can wobble by a point or two for an
  *unchanged* draft. The keep/revert logic then interprets that wobble as
  "improvement" or "regression" — so you will see drafts kept or reverted on noise,
  not real edits. A larger / hosted model scores more steadily.
- **Equal is discarded.** The keep rule requires a *strictly* better score
  (`min_improvement_delta` is `None` ⇒ the delta must exceed `0.0`). A draft that
  re-scores identically is reverted and counts toward stagnation. A genuinely good
  draft can therefore be abandoned simply because the judge could not push the
  number higher.
- **Stagnation can halt early.** With `MAX_STAGNATION = 2`, two unlucky judge calls
  in a row end the run — even mid-improvement. That is the strategy working as
  designed (it cannot tell "stuck" from "unlucky"), not a bug.
- **No PASS to chase.** Because there is no success condition, a run that climbs to
  `28/30` and a run that plateaus at `12/30` both terminate the same way — on
  stagnation or budget. The *score* is the outcome; the *halt reason* is just how
  it stopped.

These are the real properties of optimization-style loops. The point of the example
is to make them visible, not to hide them behind a tidy PASS.

The maintainer-verified known-good model is **`gemma4:31b-cloud`** (an Ollama
*cloud* model): it gives a steadier climb and, because it's served through Ollama,
needs **no code edit** — just set `SPORE_OLLAMA_MODEL=gemma4:31b-cloud`. As a
secondary alternative, to target a hosted model swap the two
`OllamaModelInterface.with_base_url(...)` constructions in `main.py` for the hosted
model interface spore-core ships. This example wires Ollama by default to keep it
key-free and to honestly exercise the strategy under stress.

## Prerequisites

- [Ollama](https://ollama.com) running locally:
  ```sh
  ollama serve &
  ollama pull llama3.2     # or a stronger model, e.g. qwen2.5-coder:7b
  ```

> This example needs a running model (Ollama by default). It is **not** part of
> the lint/format gate, which is all that CI checks — the same bar as 09.

## Run it

```sh
uv run main.py                              # llama3.2, 6-iteration budget
uv run main.py --max-iterations 8           # widen the budget
uv run main.py --model qwen2.5-coder:7b     # a stronger local model
uv run main.py --prompt "…custom task…"     # override the task
```

Or via env (`.env.example`): `SPORE_OLLAMA_MODEL`, `SPORE_OLLAMA_BASE_URL`,
`SPORE_MAX_ITERATIONS`.

## CLI flags

| Flag                   | Env                    | Default          | Meaning                                       |
|------------------------|------------------------|------------------|-----------------------------------------------|
| `--model <id>`         | `SPORE_OLLAMA_MODEL`   | `llama3.2`       | Ollama model for builder **and** judge        |
| `--max-iterations <n>` | `SPORE_MAX_ITERATIONS` | `6`              | Iteration budget → `max_turns` (NOT a target) |
| `--prompt <text>`      | —                      | the ironwood README task | Override the task the agent solves    |
| (base url)             | `SPORE_OLLAMA_BASE_URL`| `http://localhost:11434` | Ollama endpoint                       |

## Where to look

- `main.py` — the whole example. Start at `main`, then read `ReadmeQualityEvaluator`
  (the scoring oracle) and `ReportingObservability` (the per-iteration decision
  printer).
- `python/packages/spore_core/src/spore_core/harness.py` — `LoopStrategyHillClimbing`
  and the HillClimbing run loop.
- `python/packages/spore_core/src/spore_core/metric.py` — the `MetricEvaluator`
  Protocol and `MetricResult`.
- `examples/rust/10-hill-climbing` — the reference implementation.
