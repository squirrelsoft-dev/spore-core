# 09 — Self-verifying (a quality loop as a harness strategy)

The first example to wire the **`SelfVerifying`** loop strategy. The agent drafts
a Python function, a *separate evaluator run* critiques that draft against an
explicit five-point spec, and the loop **revises until it passes** or a
configurable iteration cap is hit.

The thesis: **quality loops are a harness concern, not application logic.** You
do not write a `while not good_enough: revise()` loop. You wire a strategy, a
`Verifier`, and an evaluator agent — and the harness runs the draft → critique →
revise cycle for you. The application code is one `Task` with
`LoopStrategySelfVerifying` and a verifier.

## What it demonstrates

- `LoopStrategySelfVerifying` — the build/evaluate/verdict cycle, run by the harness.
- The verify → revise loop made **visible** through the verifier seam.
- What "passes" looks like: the evaluator's own judgment against the original spec.
- A real-world stress test of the strategy against a live model — including the
  rough edges (documented honestly below).

## The task the agent must solve

Write a function `parse_int_list` that parses comma-separated integers. The
evaluator checks **five** criteria, each explicitly:

| # | Criterion       | Detail                                                                                  |
|---|-----------------|-----------------------------------------------------------------------------------------|
| 1 | Signature       | `def parse_int_list(text: str) -> list[int]` plus a custom `ParseIntListError(Exception)` |
| 2 | Edge cases      | empty/whitespace-only → `[]`; whitespace tolerated; bad token → `ParseIntListError`, never a bare crash |
| 3 | Docstring       | a docstring on the function                                                             |
| 4 | No crashes      | the `int()` parse is wrapped so a bad token becomes the typed error                     |
| 5 | Usage example   | at least one `>>>` doctest line (or inline example) in the docstring                    |

The spec lives in **one** place — the `Task` instruction — so the build agent and
the evaluator see the exact same five criteria.

### The Python idiom for the function under test

The Rust reference returns `Result<Vec<i32>, ParseIntListError>`. Python has no
`Result` type in the standard idiom, so this port asks for the equivalent: a
function returning `list[int]` on success and **raising** a custom
`ParseIntListError(Exception)` on a bad token. The "no panics" criterion becomes
"wrap the `int()` parse so a bad token becomes a typed `ParseIntListError`, not an
unhandled `ValueError`." This is what an idiomatic-Python evaluator can actually
check by reading the source.

## The verify → revise mechanism

```text
        ┌─────────────────────────────────────────────┐
        │  iteration 0..max_iterations                 │
        │                                              │
   ┌──> │  1. BUILD  (ReAct sub-loop, build agent)     │
   │    │     writes parse_int_list.py via write_file  │
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

```python
inner = EvaluatorResponseVerifier(
    pass_pattern=r"(?im)^\s*PASS\s*$",   # pass pattern
    fail_pattern=r"(?im)FAIL:\s*.+",     # fail pattern (reason captured)
    max_iterations=max_iterations,
)
verifier = ReportingVerifier(inner, max_iterations)

harness = (
    HarnessBuilder.conversational(model)
    .sandbox(sandbox)
    .tool(StandardTools.write_file())    # builder writes the draft
    .tool(StandardTools.read_file())     # evaluator reads it back
    .system_prompt(SYSTEM_PROMPT)
    .evaluator_agent(evaluator_agent)    # the evaluate-phase agent
    .verifier(verifier)                  # the oracle
    .build()
)

task = Task.new(
    prompt, new_session_id(), LoopStrategySelfVerifying(),
    budget=BudgetLimits(max_turns=12),
)

await harness.run(HarnessRunOptions(task))   # no loop code here
```

The construction API (so it mirrors the other language ports):

- **Verifier**: `EvaluatorResponseVerifier(pass_pattern, fail_pattern, max_iterations)`
  pattern-matches the evaluator's text. `PASS` → `Passed`; `FAIL: <reason>` →
  `Failed{reason}`; **neither → `Failed` by contract** (default-to-FAIL, not
  configurable — reinforced by the harness's built-in evaluator directive "you
  did NOT write this code; default to FAIL unless you can confirm it is right").
- **Wiring**: `HarnessBuilder.verifier(verifier)` and
  `HarnessBuilder.evaluator_agent(agent)`. The strategy halts as
  `HaltReasonSelfVerifyMisconfigured` if the verifier is missing.
- The harness reads `max_iterations()` off the **outer** verifier, so
  `ReportingVerifier` returns the same cap it was constructed with.

Both `EvaluatorResponseVerifier` and `ReportingVerifier` satisfy the `Verifier`
`Protocol` **structurally** — no inheritance, per the Python conventions.

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
text straight to the evaluator. **Reading the source (`_run_self_verifying` /
`_run_evaluate_phase`) shows it does not**: the evaluate phase builds a *fresh*
evaluator run whose context is seeded only with a directive containing the
`task.instruction` plus a read-only sandbox. The build agent's draft text is
**not** auto-injected into the evaluator's context. For the evaluator to actually
read the draft, the draft has to live on disk.

So this example wires the **minimal file tool set** and nothing else:

- `write_file` — the build agent saves its draft to `workspace/parse_int_list.py`.
- `read_file`  — the evaluator reads it back (its `write_file` is blocked by the
  internally-derived `ReadOnlySandbox`).

No `web_search`, no shell. The loop is still the point — these two tools exist
only to carry the draft across the build/evaluate boundary that the strategy
itself defines.

## Contrast with earlier examples

|              | 06 / 08 (ReAct / PlanExecute)              | 09 — self-verifying                                 |
|--------------|--------------------------------------------|-----------------------------------------------------|
| Strategy     | `ReAct` / `PlanExecute`                    | **`SelfVerifying`**                                 |
| Loop you write | none (single pass / planned subtasks)    | **none** — but the harness now *re-runs* on failure |
| Quality gate | n/a                                        | a `Verifier` decides pass/fail each iteration       |
| Second agent | n/a                                        | an **evaluator** run critiques the builder's draft  |
| Visibility   | stream events / lifecycle hooks            | the **verifier** seam (sub-streams are suppressed)  |

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
- **The builder describing the write in prose — mostly handled.** If the build
  agent describes the action in prose instead of calling `write_file`, the
  `conversational` preset's **adaptive prompt-based tool-calling repair** (now in
  all four cores) detects the action-intent prose, nudges the model once, and
  escalates to prompt-based `<tool_call>` markers that are parsed back into native
  tool calls — so the write lands automatically. The only honest caveat: it's a
  **one-shot** escalation — a model that *keeps* answering in prose even after the
  nudge still won't write the file. Rarely bites with a capable model.
- **Exhaustion is a normal outcome here.** A `HaltReasonSelfVerifyExhausted`
  failure after N iterations is the strategy working as designed — it bounds the
  loop. The example prints the last draft and last failure reason so you can see
  how close it got (the process exits non-zero).

The maintainer-verified known-good model is **`gemma4:31b-cloud`** (an Ollama
*cloud* model): it gives a clean demo and, because it's served through Ollama,
needs **no code edit** — just set `SPORE_OLLAMA_MODEL=gemma4:31b-cloud`. As a
secondary alternative, to target a hosted model swap the two
`OllamaModelInterface.with_base_url(...)` constructions in `main.py` for the
hosted model interface spore-core ships. This example wires Ollama by default to
keep it key-free and to honestly exercise the strategy under stress.

## Prerequisites

- [Ollama](https://ollama.com) running locally:
  ```sh
  ollama serve &
  ollama pull llama3.2     # or a stronger coder model, e.g. qwen2.5-coder:7b
  ```

## Run it

```sh
uv run main.py                              # llama3.2, 3 iterations
uv run main.py --max-iterations 5           # raise the cap
uv run main.py --model qwen2.5-coder:7b     # a stronger local model
uv run main.py --prompt "…custom spec…"     # override the task
```

Or via env (`.env.example`): `SPORE_OLLAMA_MODEL`, `SPORE_OLLAMA_BASE_URL`,
`SPORE_MAX_ITERATIONS`.

## CLI flags

| Flag                   | Env                    | Default          | Meaning                                    |
|------------------------|------------------------|------------------|--------------------------------------------|
| `--model <id>`         | `SPORE_OLLAMA_MODEL`   | `llama3.2`       | Ollama model for builder **and** evaluator |
| `--max-iterations <n>` | `SPORE_MAX_ITERATIONS` | `3`              | Verify→revise cap (visible in output)      |
| `--prompt <text>`      | —                      | the 5-point spec | Override the task the agent solves         |
| (base url)             | `SPORE_OLLAMA_BASE_URL`| `http://localhost:11434` | Ollama endpoint                   |
