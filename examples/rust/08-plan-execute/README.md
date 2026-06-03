# 08 — Plan-and-execute (multi-step goal decomposition)

The first example to swap the **loop strategy**. The agent is given a multi-step
goal — *research three Rust async runtimes and write a cited comparison to a
file* — and instead of reacting one tool call at a time, it **decomposes the goal
into a plan first**, prints that plan, then executes each subtask in turn.

The thesis: **swapping the loop strategy is a one-line harness change.** Same
builder, same sandbox, same tools as [`06`](../06-web-research). Only the
`LoopStrategy` on the `Task` changes — from `ReAct` to `PlanExecute` — and the
behavior changes qualitatively: the agent reasons about its own task structure
before touching any tool.

## The contrast with 06

|          | 06 — web-research                                  | 08 — plan-execute                                              |
| -------- | -------------------------------------------------- | -------------------------------------------------------------- |
| Builder  | `conversational(model)`                            | `conversational(model)` *(same)*                               |
| Sandbox  | `WorkspaceScopedSandbox` over `workspace/`         | `WorkspaceScopedSandbox` over `workspace/` *(same)*            |
| Tools    | `web_search` + `write_file` + `read_file`          | `web_search` + `write_file` + `read_file` *(same)*             |
| Strategy | `ReAct { max_iterations: 10 }`                     | **`PlanExecute { plan_model: None }`** *(the swap)*            |
| Behavior | reacts step-by-step, no upfront plan               | **prints a full plan first, then runs each subtask**           |
| Output   | stream-printed `think` / `act` / `obs`             | a `── plan ──` banner + `[i/N]` subtask lines (via hooks)      |
| Side eff | writes `answer.md`                                 | writes `async-comparison.md`                                   |

Everything above the strategy row is held constant on purpose. The single
substantive change is the loop strategy:

```rust
// 06 — react step-by-step:
let task = Task::new(prompt, SessionId::generate(),
    LoopStrategy::ReAct { max_iterations: 10 });

// 08 — decompose the goal first, then execute each subtask:
let task = Task::new(prompt, SessionId::generate(),
    LoopStrategy::PlanExecute { plan_model: None })
    .with_budget(BudgetLimits { max_turns: Some(24), ..Default::default() });
```

With `PlanExecute`, the harness runs one constrained **planner turn** first. The
planner must return strict JSON `{ "tasks": [...], "rationale": ... }`; that text
is captured into a `PlanArtifact`, the plan is surfaced, and then each subtask
runs in its own bounded sub-loop. The turn budget is **divided across subtasks**
(per-task cap = remaining_turns / remaining_tasks), so we set a generous
`max_turns`.

## Watching the plan — via lifecycle hooks, not stream events

There are **no plan/subtask stream events**. The plan becomes visible through the
hook chain. This example registers a `PlanExecuteReporter` (`impl Hook`) on two
events:

- **`OnPlanCreated { plan: &mut PlanArtifact }`** — fires once, after the plan is
  captured and before any subtask runs. We print the rationale and numbered tasks.
- **`OnTaskAdvance { task: &mut Task, task_index, total_tasks }`** — fires before
  each subtask. We print `[i/N] <instruction>`.

```rust
let chain = StandardHookChain::new();
chain.register(Arc::new(PlanExecuteReporter))?;
let harness = HarnessBuilder::conversational(model)
    /* same sandbox + tools as 06 */
    .hooks(Arc::new(chain))
    .build();
```

Sample output (the money moment, then execution):

```
── plan ──
rationale: Compare three runtimes across performance, maturity, and use cases, then write the file.
  1. Search the web for tokio performance and ecosystem maturity
  2. Search the web for async-std performance and ecosystem maturity
  3. Search the web for smol performance and ecosystem maturity
  4. Synthesize the findings into a comparison and write async-comparison.md
──────────

[1/4] Search the web for tokio performance and ecosystem maturity
[2/4] Search the web for async-std performance and ecosystem maturity
[3/4] Search the web for smol performance and ecosystem maturity
[4/4] Synthesize the findings into a comparison and write async-comparison.md

answer (N turn(s)): I researched the three runtimes and wrote async-comparison.md …

plan persisted in extras["plan_execute"] with 4 subtask(s)

async-comparison.md now exists on disk: …/workspace/async-comparison.md
```

The plan is also persisted to `session_state.extras["plan_execute"]`
(`PLAN_EXECUTE_EXTRAS_KEY`); the example reads it back on success to confirm.

### The agent writes the file, not `main`

Just like 04 and 06, the file write happens **inside the loop** via the catalogue
`write_file` tool, backed by the `WorkspaceScopedSandbox`. `main.rs` never writes
`async-comparison.md` itself — the agent does, and the sandbox keeps it inside
`workspace/`.

## The search backend (and an honesty note about #108)

There is **no live web-search backend in spore-core**. The endpoint is injected,
so you must supply one. The example reads it from `SPORE_WEB_SEARCH_ENDPOINT` and
exits if it is unset. `web_search` POSTs the query as JSON `{ "query": ... }` and
hands the response body back to the agent verbatim.

Any endpoint that accepts that shape works:

- a self-hosted **SearXNG** JSON endpoint, or
- a small **mock** you run locally for the demo.

**Raw Brave / Tavily are not yet drop-in.** They require a custom auth header
(`X-Subscription-Token` / `Authorization`) that the current `web_search` tool
does not send. That gap is tracked as core
[issue #108](https://github.com/squirrelsoft-dev/spore-core/issues/108); this
example deliberately does **not** ship a local proxy/adapter to paper over it.

## An honesty note about PlanExecute

`PlanExecute` is **largely untested against live models** and has real rough
edges you should know about before running:

- The planner must return **strict JSON** `{ "tasks": [...], "rationale": ... }`
  (a single ```` ``` ```` / ```` ```json ```` fence is tolerated and stripped).
  Anything else fails the run with `UnparseablePlan`.
- An **empty `tasks` array** halts with `HaltEmptyPlan`.
- A **subtask error aborts the whole run** with `HaltStepFailed` — there is no
  per-step retry.
- The turn **budget is divided across subtasks**, so a stingy `max_turns` starves
  later steps. Set it generously (this example uses `24`).
- **Small local models (e.g. `llama3.2`) often garble the plan JSON.** A larger
  hosted model produces a cleaner, more reliable demo. The harness is
  model-agnostic — swap the model interface and change nothing else.
- Subtask **inner tool calls stream in the TypeScript port only**; the Rust
  harness suppresses the sub-loop stream, so the `OnPlanCreated` / `OnTaskAdvance`
  hooks are the portable, cross-language view of execution.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
# A {"query"}->JSON search endpoint (SearXNG or a local mock):
export SPORE_WEB_SEARCH_ENDPOINT=http://localhost:8888/search
```

See `.env.example` for all the variables.

## Run

```sh
cd examples/rust/08-plan-execute
cargo run
cargo run -- --prompt "Compare three Rust web frameworks (axum, actix-web, rocket) on performance, ergonomics, and ecosystem; cite sources and save to async-comparison.md."
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint,
and `--model` overrides the id for a single run — same as 04 / 05 / 06.

`async-comparison.md` is written into `workspace/` and is `.gitignore`d so re-runs
stay clean.
