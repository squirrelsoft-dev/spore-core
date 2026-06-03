# 09 — Self-verifying (a quality loop as a harness strategy)

The first Go example to wire the **`SelfVerifying`** loop strategy. The agent
drafts a Go function, a *separate evaluator run* critiques that draft against an
explicit five-point spec, and the loop **revises until it passes** or a
configurable iteration cap is hit.

The thesis: **quality loops are a harness concern, not application logic.** You
do not write a `for !goodEnough { revise() }` loop. You wire a strategy, a
`Verifier`, and an evaluator agent — and the harness runs the draft → critique →
revise cycle for you. The application code is one `Task` with
`LoopStrategy{Kind: StrategySelfVerifying}` and a verifier.

## What it demonstrates

- `LoopStrategy{Kind: StrategySelfVerifying}` — the build/evaluate/verdict cycle, run by the harness.
- The verify → revise loop made **visible** through the verifier seam.
- What "passes" looks like: the evaluator's own judgment against the original spec.
- A real-world stress test of the strategy against a live model — including the
  rough edges (documented honestly below).

## The task the agent must solve

Write a function `ParseIntList` that parses comma-separated integers. The
verifier checks **five** criteria, each explicitly:

| # | Criterion     | Detail                                                                                       |
|---|---------------|----------------------------------------------------------------------------------------------|
| 1 | Signature     | `func ParseIntList(input string) ([]int, error)` returning a typed/custom error on bad input |
| 2 | Edge cases    | empty/whitespace-only → empty slice, nil error; whitespace tolerated; bad token → error, never panic |
| 3 | Doc comments  | a `//` doc comment on the function                                                           |
| 4 | No panics     | no `panic(...)` and no must-style helper that panics                                         |
| 5 | Usage example | at least one usage example in the doc comment                                                |

The spec lives in **one** place — the `Task` instruction — so the build agent
and the evaluator see the exact same five criteria.

## The verify → revise mechanism

```text
        ┌─────────────────────────────────────────────┐
        │  iteration 0..maxIterations                  │
        │                                              │
   ┌──> │  1. BUILD  (ReAct sub-loop, build agent)     │
   │    │     writes parse_int_list.go via write_file  │
   │    │                                              │
   │    │  2. EVALUATE  (fresh run, evaluator agent,   │
   │    │     read-only sandbox) reads the file,       │
   │    │     critiques it, emits PASS / FAIL: <why>   │
   │    │                                              │
   │    │  3. VERIFY  Verifier turns the critique into │
   │    │     a verdict (Passed | Failed{reason})      │
   │    └─────────────────┬────────────────────────────┘
   │                      │
   │   Failed{reason} ────┘  reason injected into the
   └── next build turn; loop revises
                          │
       Passed ────────────┴──> halt with the build output (success)
       cap reached ───────────> halt SelfVerifyExhausted (failure)
```

You wire three things; the harness does the rest:

```go
inner, _ := verifier.NewEvaluatorResponseVerifier(
    `(?im)^\s*PASS\s*$`, // pass pattern
    `(?im)FAIL:\s*.+`,   // fail pattern (reason captured)
    maxIterations,
)
// Bridge the verifier-package type into the root-package harness seam, then
// wrap it so the cycle prints.
harnessVerifier := newReportingVerifier(verifier.AsHarnessVerifier(inner), maxIterations)

cfg := observability.ConversationalBuilder(mi).
    Sandbox(sandbox).
    Tool(tools.StandardTools{}.WriteFile()). // builder writes the draft
    Tool(tools.StandardTools{}.ReadFile()).  // evaluator reads it back
    SystemPrompt(systemPrompt).
    BuildConfig()
cfg.Verifier = harnessVerifier      // the oracle
cfg.EvaluatorAgent = evaluatorAgent // the evaluate-phase agent
harness := sporecore.NewStandardHarness(cfg)

task := sporecore.NewTask(prompt, sporecore.NewSessionID(),
    sporecore.LoopStrategy{Kind: sporecore.StrategySelfVerifying}).
    WithBudget(sporecore.BudgetLimits{MaxTurns: &maxTurns})

harness.Run(context.Background(), sporecore.NewHarnessRunOptions(task)) // no loop code here
```

### The construction API (so the cross-language ports stay aligned)

- **Verifier**: `verifier.NewEvaluatorResponseVerifier(passPattern, failPattern, maxIterations)`
  pattern-matches the evaluator's text. `PASS` → `Passed`; `FAIL: <reason>` →
  `Failed{reason}`; **neither → `Failed` by contract** (default-to-FAIL, not
  configurable — reinforced by the harness's built-in evaluator directive "you
  did NOT write this code; default to FAIL unless you can confirm it is right").
- **The bridge**: `verifier.AsHarnessVerifier(v)` adapts a `verifier.Verifier`
  into the root-package `sporecore.Verifier` seam. spore-core defines `Verifier`
  in root-package terms (`Verify(ctx, SelfVerifyInput) SelfVerifyVerdict` +
  `MaxIterations()`); the concrete verifiers live in the `verifier` package to
  avoid an import cycle, so the adapter is mandatory.
- **Wiring** (Go asymmetry): the observability `HarnessBuilder` has **no**
  `.Verifier()` / `.EvaluatorAgent()` setters. Build the config with
  `BuildConfig()`, then set `cfg.Verifier` and `cfg.EvaluatorAgent` before
  `NewStandardHarness(cfg)` — the same shape 08 uses for `cfg.Hooks`. The
  strategy halts as `SelfVerifyMisconfigured` if the verifier is nil.
- The harness reads `MaxIterations()` off the **outer** verifier, so
  `reportingVerifier` forwards the same cap it was constructed with.

### Watching the loop — the `reportingVerifier` seam

The build and evaluate sub-runs are streamed with a **suppressed sink** (the same
design choice as PlanExecute), so there is no per-token stream to watch. The one
reliable seam is the **verifier**: the harness calls `Verify(SelfVerifyInput)`
once per iteration, and `SelfVerifyInput` carries everything worth seeing —

- `BuildResult` → the **draft** (the function the agent wrote),
- `EvalResult`  → the **critique** (what the evaluator said),
- `Iteration`   → the 0-indexed cycle number.

`reportingVerifier` is a tiny `sporecore.Verifier` decorator: it prints a 1-based
iteration header (`iteration 2/3`), the draft, the critique, and the verdict,
then delegates the actual pass/fail decision to the bridged
`EvaluatorResponseVerifier`. That is the entire observability story for this
strategy.

## The tool decision — one file seam, not zero tools

The issue suggested keeping tools out entirely if the strategy feeds the build
text straight to the evaluator. **Reading the source (`runSelfVerifying` /
`runSelfVerifyEvaluatePhase` in `self_verifying.go`) shows it does not**: the
evaluate phase builds a *fresh* evaluator run whose context is seeded only with a
directive containing the `task.Instruction` plus a read-only sandbox. The build
agent's draft text is **not** auto-injected into the evaluator's context. For the
evaluator to actually read the draft, the draft has to live on disk.

So this example wires the **minimal file tool set** and nothing else:

- `write_file` — the build agent saves its draft to `workspace/parse_int_list.go`.
- `read_file`  — the evaluator reads it back (its `write_file` is blocked by the
  internally-derived `ReadOnlySandbox`).

No `web_search`, no shell. The loop is still the point — these two tools exist
only to carry the draft across the build/evaluate boundary that the strategy
itself defines.

## Contrast with earlier examples

|                | 06 / 08 (ReAct / PlanExecute)             | 09 — self-verifying                                 |
|----------------|-------------------------------------------|-----------------------------------------------------|
| Strategy       | `ReAct` / `PlanExecute`                   | **`SelfVerifying`**                                 |
| Loop you write | none (single pass / planned subtasks)     | **none** — but the harness now *re-runs* on failure |
| Quality gate   | n/a                                       | a `Verifier` decides pass/fail each iteration       |
| Second agent   | n/a                                       | an **evaluator** run critiques the builder's draft  |
| Visibility     | stream events / lifecycle hooks           | the **verifier** seam (sub-streams are suppressed)  |

## Rough edges (honest, because this is also a stress test)

SelfVerifying is demanding and, against a small local model, can be flaky. With
the maintainer-verified `gemma4:31b-cloud` (see below) the loop runs well in
practice; these remain risks chiefly on weaker/smaller local models:

- **The evaluator mis-judges.** Small models emit false `PASS` (rubber-stamping
  broken code) or false `FAIL` (rejecting correct code), and sometimes neither a
  clean `PASS` nor a `FAIL:` line — which the verifier (correctly) treats as
  FAIL. Expect the loop to exhaust without passing on weak models.
- **Format drift.** The evaluator must end with exactly `PASS` or `FAIL: …`.
  Models that wrap the verdict in prose or markdown can dodge the patterns. The
  patterns here are lenient (`(?im)` multiline, anchored `PASS`, `FAIL:`
  anywhere) but not bulletproof.
- **The builder may not call the tool.** If the build agent answers with code in
  prose instead of calling `write_file`, the evaluator reads a stale/empty file
  and FAILs. Note: the Go core does **not** yet have the adaptive prompt-based
  tool-call repair that the Rust core gained in #111 (porting it is a known
  follow-up), so there is no automatic recovery here. In practice a capable model
  like `gemma4:31b-cloud` rarely triggers this; small local models slip.
- **Exhaustion is a normal outcome here.** A `SelfVerifyExhausted` failure after
  N iterations is the strategy working as designed — it bounds the loop. The
  example prints the last draft and last failure reason so you can see how close
  it got, then exits non-zero.

The maintainer-verified known-good model is **`gemma4:31b-cloud`** (an Ollama
*cloud* model): it gives a clean demo and, because it's served through Ollama,
needs **no code edit** — just set `SPORE_OLLAMA_MODEL=gemma4:31b-cloud`. As a
secondary alternative, to target a hosted model swap the two
`ollama.WithBaseURL(...)` constructions in `main.go` for spore-core's hosted
model interface (e.g. the Anthropic interface). This example wires Ollama by
default to keep it key-free and to honestly exercise the strategy under stress.

## Prerequisites

- [Ollama](https://ollama.com) running locally:
  ```sh
  ollama serve &
  ollama pull llama3.2     # or a stronger coder model, e.g. qwen2.5-coder:7b
  ```

## Run it

```sh
go run .                              # llama3.2, 3 iterations
go run . --max-iterations 5           # raise the cap
go run . --model qwen2.5-coder:7b     # a stronger local model
go run . --prompt "…custom spec…"     # override the task
```

Or via env (`.env.example`): `SPORE_OLLAMA_MODEL`, `SPORE_OLLAMA_BASE_URL`,
`SPORE_MAX_ITERATIONS`.

## CLI flags

| Flag                   | Env                    | Default          | Meaning                                     |
|------------------------|------------------------|------------------|---------------------------------------------|
| `--model <id>`         | `SPORE_OLLAMA_MODEL`   | `llama3.2`       | Ollama model for builder **and** evaluator  |
| `--max-iterations <n>` | `SPORE_MAX_ITERATIONS` | `3`              | Verify→revise cap (visible in output)       |
| `--prompt <text>`      | —                      | the 5-point spec | Override the task the agent solves          |
| (base url)             | `SPORE_OLLAMA_BASE_URL`| `http://localhost:11434` | Ollama endpoint                     |
