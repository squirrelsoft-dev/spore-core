# Loop strategies

> Language-agnostic — no code. See [architecture](./architecture.md) for how the loop fits in.

The harness drives the loop; the agent executes one turn. A **loop strategy** determines the
outer structure — how many turns, what happens between them, and what "done" means. spore-core
ships five.

## ReAct

The foundational pattern. Thought / Action / Observation interleave in the context: the model
thinks, requests a tool, the harness runs it and appends the result, the model observes and
thinks again. The loop runs until the model returns a final response with no tool calls, at
which point the termination policy decides whether that response actually counts as done.

Use it for: almost everything. It's the default shape, and the other strategies are
specializations of it. A turn cap (`max_iterations`) bounds runaway loops.

## PlanExecute

Two phases. A **planner** model produces a plan artifact once; an **executor** model then works
the steps of that plan in a loop. The two phases can use different models — a stronger model to
plan, a cheaper one to execute. The plan artifact is preserved in session state so callers can
inspect or resume against it.

Use it for: multi-step tasks where committing to a plan up front improves reliability, or where
you want to review the plan before any action is taken.

## SelfVerifying

A loop within a loop. A **build** phase runs until the agent claims it is done. Then an
**evaluate** phase runs a *separate evaluator harness* with three guarantees:

- a **read-only sandbox** — no write or edit tools,
- a **fresh session context** — it never shares a session with the builder, and
- an explicit evaluator role: *"You are a fresh evaluator. You did not write the code you are
  reviewing."*

This is the **Default-FAIL contract**: the evaluator cannot be biased by having watched the work
happen. If it finds problems, the findings are injected back into the build context and the build
loop continues. Authoring and review are kept in separate lanes by construction.

Use it for: work where correctness matters more than speed and a second, unbiased pass pays for
itself.

## HillClimbing

An iterative optimization loop. It establishes a baseline metric, proposes a change, evaluates
the metric, and **keeps or reverts** based on improvement. Knobs:

- `max_stagnation` — halt after N consecutive non-improvements (`None` = run until stopped),
- `revert_on_no_improvement` — reset (e.g. via git) if the metric didn't improve,
- `min_improvement_delta` — the smallest change that counts as progress.

The harness — not the agent — writes the improvement record (a results log per task), so the
optimization history is trustworthy.

Use it for: optimization tasks with a measurable objective — performance tuning, eval-score
climbing, prompt or parameter search.

## Ralph

A continuation loop for work that spans **multiple context windows**. Ralph intercepts the
model's attempt to exit, resets the context window, reloads state from the filesystem (a progress
file, the git log, a feature list), and continues until an external completion check passes. The
filesystem is the handoff medium — it's what makes work larger than a single context window
possible.

In spore-core this is driven by a `.spore/progress.json` file: while it reports incomplete work
the loop spins up a fresh context window; when it reports complete, the loop terminates
successfully. When the progress file is absent, the Ralph machinery is inert and does not
interfere with the other strategies.

Use it for: long-running, multi-session jobs — large migrations, "build the whole feature list,"
anything that won't fit in one window.

## Choosing

| If you want… | Use |
|--------------|-----|
| The default think-act-observe loop | **ReAct** |
| A plan committed up front, then executed | **PlanExecute** |
| An unbiased second pass before "done" | **SelfVerifying** |
| To optimize against a measurable metric | **HillClimbing** |
| Work that outgrows a single context window | **Ralph** |

Termination is always the [termination policy's](../reference/harness-builder.md) call, evaluated
against external state — the strategy shapes the loop, but it never lets the model declare victory
unchecked.
