# 09 — Self-verifying (a quality loop as a harness strategy)

The first example to wire the **`SelfVerifying`** loop strategy. The agent drafts
a TypeScript function, a *separate evaluator run* critiques that draft against an
explicit five-point spec, and the loop **revises until it passes** or a
configurable iteration cap is hit.

The thesis: **quality loops are a harness concern, not application logic.** You
do not write a `while (!goodEnough) { revise() }` loop. You wire a strategy, a
`Verifier`, and an evaluator agent — and the harness runs the draft → critique →
revise cycle for you. The application code is one `Task` with
`{ kind: "self_verifying" }` and a verifier.

## What it demonstrates

- `LoopStrategy.self_verifying` — the build/evaluate/verdict cycle, run by the harness.
- The verify → revise loop made **visible** through the verifier seam.
- What "passes" looks like: the evaluator's own judgment against the original spec.
- A real-world stress test of the strategy against a live model — including the
  rough edges (documented honestly below).

## The task the agent must solve

Write a function `parseIntList` that parses comma-separated integers. The result
is an idiomatic discriminated union —
`{ ok: true; value: number[] } | { ok: false; error: ParseIntListError }` — so
failure is a typed value, never an unexpected throw. The verifier checks **five**
criteria, each explicitly:

| # | Criterion     | Detail                                                                                                  |
|---|---------------|---------------------------------------------------------------------------------------------------------|
| 1 | Signature     | `export function parseIntList(input: string): ParseIntListResult` (discriminated union + custom error)  |
| 2 | Edge cases    | empty/whitespace-only → `{ ok: true, value: [] }`; whitespace tolerated; bad token → error variant, no throw |
| 3 | Doc comments  | a JSDoc block on the function                                                                            |
| 4 | No crashers   | no non-null assertions (`!`) or unchecked casts that can crash                                           |
| 5 | Usage example | an `@example` block in the JSDoc                                                                         |

The spec lives in **one** place — the `Task` instruction — so the build agent and
the evaluator see the exact same five criteria.

### Why a discriminated result, not `throw`?

The Rust reference returns `Result<Vec<i32>, ParseIntListError>`. The idiomatic
TypeScript analogue of a `Result` is a discriminated union the caller pattern-
matches on (`result.ok ? result.value : result.error`), so the same "errors are
typed values, never an unexpected crash" semantics carry across both languages,
and the evaluator can check the signature criterion by looking for the union.

## The verify → revise mechanism

```text
        ┌─────────────────────────────────────────────┐
        │  iteration 0..max_iterations                 │
        │                                              │
   ┌──> │  1. BUILD  (ReAct sub-loop, build agent)     │
   │    │     writes parse-int-list.ts via write_file  │
   │    │                                              │
   │    │  2. EVALUATE  (fresh run, evaluator agent,   │
   │    │     read-only sandbox) reads the file,       │
   │    │     critiques it, emits PASS / FAIL: <why>   │
   │    │                                              │
   │    │  3. VERIFY  Verifier turns the critique into │
   │    │     a verdict (passed | failed{reason})      │
   │    └─────────────────┬────────────────────────────┘
   │                      │
   │   failed{reason} ────┘  reason injected into the
   └── next build turn; loop revises
                          │
       passed ────────────┴──> halt with the build output (success)
       cap reached ───────────> halt self_verify_exhausted (failure)
```

You wire three things; the harness does the rest:

```ts
const innerVerifier = new verifier.EvaluatorResponseVerifier({
  pass_pattern: "(?im)^\\s*PASS\\s*$", // pass pattern
  fail_pattern: "(?im)FAIL:\\s*.+", // fail pattern (reason captured)
  max_iterations: maxIterations,
});
const reportingVerifier = new ReportingVerifier(innerVerifier, maxIterations);

const harness = HarnessBuilder.conversational(model)
  .sandbox(sandbox)
  .tool(StandardTools.writeFile()) // builder writes the draft
  .tool(StandardTools.readFile()) // evaluator reads it back
  .systemPrompt(SYSTEM_PROMPT)
  .evaluatorAgent(evaluatorAgent) // the evaluate-phase agent
  .verifier(reportingVerifier) // the oracle
  .build();

const task = newTask(
  prompt,
  SessionId.generate(),
  { kind: "self_verifying" },
  { max_turns: 12 },
);

await harness.run({ task }); // no loop code here
```

The construction API (so the other language ports can mirror it):

- **Verifier**: `new EvaluatorResponseVerifier({ pass_pattern, fail_pattern, max_iterations })`
  pattern-matches the evaluator's text. `PASS` → `passed`; `FAIL: <reason>` →
  `failed{reason}`; **neither → `failed` by contract** (default-to-FAIL, not
  configurable — reinforced by the harness's built-in evaluator directive "you
  did NOT write this code; default to FAIL unless you can confirm it is right").
  It lives under the `verifier` namespace export (`verifier.EvaluatorResponseVerifier`).
- **Pattern flags**: JavaScript has no inline `(?im)` group, but
  `EvaluatorResponseVerifier` accepts the same Rust-fixture pattern strings the
  other ports use — it strips the leading inline-flag group and sets the JS `i`/`m`
  flags — so the patterns are identical across languages.
- **Wiring**: `HarnessBuilder.verifier(Verifier)` and
  `HarnessBuilder.evaluatorAgent(Agent)`. The strategy halts as
  `self_verify_misconfigured` if the verifier is missing.
- The harness reads `maxIterations()` off the **outer** verifier, so
  `ReportingVerifier` returns the same cap it was constructed with.

### Watching the loop — the `ReportingVerifier` seam

The build and evaluate sub-runs are streamed with a **suppressed sink** (the same
design choice as PlanExecute), so there is no per-token stream to watch. The one
reliable seam is the **verifier**: the harness calls `verify(input)` once per
iteration, and `VerifierInput` carries everything worth seeing —

- `build_result` → the **draft** (the function the agent wrote),
- `eval_result`  → the **critique** (what the evaluator said),
- `iteration`    → the 0-indexed cycle number.

`ReportingVerifier` is a tiny `Verifier` decorator: it prints a 1-based iteration
header (`iteration 2/3`), the draft, the critique, and the verdict, then delegates
the actual pass/fail decision to the inner `EvaluatorResponseVerifier`. That is
the entire observability story for this strategy.

## The tool decision — one file seam, not zero tools

The issue suggested keeping tools out entirely if the strategy feeds the build
text straight to the evaluator. **Reading the source (`runSelfVerifying` /
`runEvaluatePhase` in `packages/core/src/harness/standard.ts`) shows it does
not**: the evaluate phase builds a *fresh* evaluator run whose context is seeded
only with a directive containing the task instruction plus a read-only sandbox.
The build agent's draft text is **not** auto-injected into the evaluator's
context. For the evaluator to actually read the draft, the draft has to live on
disk.

So this example wires the **minimal file tool set** and nothing else:

- `write_file` — the build agent saves its draft to `workspace/parse-int-list.ts`.
- `read_file`  — the evaluator reads it back (its `write_file` is blocked by the
  internally-derived read-only sandbox).

No `web_search`, no shell. The loop is still the point — these two tools exist
only to carry the draft across the build/evaluate boundary that the strategy
itself defines.

## Contrast with earlier examples

|                | 06 / 08 (ReAct / PlanExecute)            | 09 — self-verifying                                 |
|----------------|------------------------------------------|------------------------------------------------------|
| Strategy       | `re_act` / `plan_execute`                | **`self_verifying`**                                |
| Loop you write | none (single pass / planned subtasks)    | **none** — but the harness now *re-runs* on failure  |
| Quality gate   | n/a                                      | a `Verifier` decides pass/fail each iteration        |
| Second agent   | n/a                                      | an **evaluator** run critiques the builder's draft   |
| Visibility     | stream events / lifecycle hooks          | the **verifier** seam (sub-streams are suppressed)   |

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
  patterns here are lenient (`(?im)` multiline, anchored `PASS`, `FAIL:` anywhere)
  but not bulletproof.
- **The builder may not call the tool.** If the build agent answers with code in
  prose instead of calling `write_file`, the evaluator reads a stale/empty file
  and FAILs. Note: the TypeScript core does **not** yet have the adaptive
  prompt-based tool-call repair that the Rust core gained in #111 (porting it is a
  known follow-up), so there is no automatic recovery here. In practice a capable
  model like `gemma4:31b-cloud` rarely triggers this; small local models slip.
- **Exhaustion is a normal outcome here.** A `self_verify_exhausted` failure after
  N iterations is the strategy working as designed — it bounds the loop. The
  example prints the last draft and last failure reason so you can see how close
  it got.

The maintainer-verified known-good model is **`gemma4:31b-cloud`** (an Ollama
*cloud* model): it gives a clean demo and, because it's served through Ollama,
needs **no code edit** — just set `SPORE_OLLAMA_MODEL=gemma4:31b-cloud`. As a
secondary alternative, spore-core ships an `AnthropicModelInterface`; to target
it, swap the two `OllamaModelInterface.withBaseUrl(...)` constructions in
`src/main.ts` for an `AnthropicModelInterface` instance. This example wires Ollama
by default to keep it key-free and to honestly exercise the strategy under stress.

## Prerequisites

- [Ollama](https://ollama.com) running locally:
  ```sh
  ollama serve &
  ollama pull llama3.2     # or a stronger coder model, e.g. qwen2.5-coder:7b
  ```
- `pnpm install` (from this directory) to link the local `@spore/core` and
  `@spore/tools` workspace packages.

## Run it

```sh
pnpm install
pnpm start                                # llama3.2, 3 iterations
pnpm start -- --max-iterations 5          # raise the cap
pnpm start -- --model qwen2.5-coder:7b    # a stronger local model
pnpm start -- --prompt "…custom spec…"    # override the task
```

Or via env (`.env.example`): `SPORE_OLLAMA_MODEL`, `SPORE_OLLAMA_BASE_URL`,
`SPORE_MAX_ITERATIONS`.

## CLI flags

| Flag                   | Env                    | Default          | Meaning                                    |
|------------------------|------------------------|------------------|--------------------------------------------|
| `--model <id>`         | `SPORE_OLLAMA_MODEL`   | `llama3.2`       | Ollama model for builder **and** evaluator |
| `--max-iterations <n>` | `SPORE_MAX_ITERATIONS` | `3`              | Verify→revise cap (visible in output)      |
| `--prompt <text>`      | —                      | the 5-point spec | Override the task the agent solves         |
| (base url)             | `SPORE_OLLAMA_BASE_URL`| `http://localhost:11434` | Ollama endpoint                    |
