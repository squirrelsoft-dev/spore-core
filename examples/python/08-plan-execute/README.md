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

|          | 06 — web-research                          | 08 — plan-execute                                          |
| -------- | ------------------------------------------ | --------------------------------------------------------- |
| Builder  | `conversational(model)`                    | `conversational(model)` *(same)*                          |
| Sandbox  | `WorkspaceScopedSandbox` over `workspace/` | `WorkspaceScopedSandbox` over `workspace/` *(same)*       |
| Tools    | `web_search` + `write_file` + `read_file`  | `web_search` + `write_file` + `read_file` *(same)*        |
| Strategy | `LoopStrategyReAct(max_iterations=10)`     | **`LoopStrategyPlanExecute()`** *(the swap)*              |
| Behavior | reacts step-by-step, no upfront plan       | **prints a full plan first, then runs each subtask**      |
| Output   | stream-printed `think` / `act` / `obs`     | a `── plan ──` banner + `[i/N]` subtask lines (via hooks) |
| Side eff | writes `answer.md`                         | writes `async-comparison.md`                              |

Everything above the strategy row is held constant on purpose. The single
substantive change is the loop strategy:

```python
# 06 — react step-by-step:
task = Task.new(prompt, new_session_id(), LoopStrategyReAct(max_iterations=10))

# 08 — decompose the goal first, then execute each subtask:
task = Task.new(
    prompt,
    new_session_id(),
    LoopStrategyPlanExecute(),
    budget=BudgetLimits(max_turns=64),
)
```

With `PlanExecute`, the harness runs one constrained **planner turn** first. The
planner must return strict JSON `{ "tasks": [...], "rationale": ... }`; that text
is captured into a `PlanArtifact`, the plan is surfaced, and then each subtask
runs in its own bounded sub-loop. The turn budget is **divided across subtasks**
(per-task cap = remaining_turns / remaining_tasks), so we set a generous
`max_turns`.

## Watching the plan — via lifecycle hooks, not stream events

There are **no plan/subtask stream events**. The plan becomes visible through the
hook chain. This example registers a `PlanExecuteReporter` (a `Hook`) on two
events:

- **`OnPlanCreated`** (`OnPlanCreatedContext`) — fires once, after the plan is
  captured and before any subtask runs. We print the rationale and numbered tasks.
- **`OnTaskAdvance`** (`OnTaskAdvanceContext`) — fires before each subtask,
  carrying the full `task` plus `task_index` (0-based) and `total_tasks`. We
  print `[i/N] <instruction>`.

```python
chain = StandardHookChain()
chain.register(PlanExecuteReporter())
harness = (
    HarnessBuilder.conversational(model)
    # same sandbox + tools as 06
    .hooks(chain)
    .build()
)
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
`write_file` tool, backed by the `WorkspaceScopedSandbox`. `main.py` never writes
`async-comparison.md` itself — the agent does, and the sandbox keeps it inside
`workspace/`.

## The search backend

There is **no live web-search backend in spore-core**. The endpoint is injected,
so you must supply one. The example reads it from `SPORE_WEB_SEARCH_ENDPOINT` and
exits if it is unset. `web_search` issues `GET <endpoint>?q=<query>` and hands the
response body back to the agent verbatim — wired identically to
[`06`](../06-web-research) via `WebSearchTool.with_config`.

This example targets a self-hosted **SearXNG** instance, whose JSON API is
`GET /search?q=<query>&format=json`. You put the `?format=json` on the endpoint
URL itself; the GET path **preserves** that existing query string and appends
`q=<query>`. The configurable GET method + query param this relies on were added
in [#108](https://github.com/squirrelsoft-dev/spore-core/issues/108) (now
resolved).

### SearXNG setup

1. Enable the JSON output format in your SearXNG `settings.yml`:

   ```yaml
   search:
     formats:
       - html
       - json
   ```

2. Restart SearXNG so the change takes effect.
3. Point the example at it (note the `?format=json`):

   ```sh
   export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
   ```

## An honesty note about PlanExecute

`PlanExecute` is **largely untested against live models** and has real rough
edges you should know about before running:

- The planner must return **strict JSON** `{ "tasks": [...], "rationale": ... }`.
  Anything else fails the run with `UnparseablePlan`.
- An **empty `tasks` array** halts with `HaltEmptyPlan`.
- A **subtask error aborts the whole run** with `HaltStepFailed` — there is no
  per-step retry.
- The turn **budget is divided across subtasks**, so a stingy `max_turns` starves
  later steps. Set it generously (this example uses `24`).
- **Small local models (e.g. `llama3.2`) often garble the plan JSON.** A larger
  hosted model produces a cleaner, more reliable demo. The harness is
  model-agnostic — swap the model interface and change nothing else.
- Subtask **inner tool calls stream in the TypeScript port only**; the Python
  harness (like Rust and Go) suppresses the sub-loop stream, so the
  `OnPlanCreated` / `OnTaskAdvance` hooks are the portable, cross-language view of
  execution.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
# A SearXNG JSON endpoint (see "SearXNG setup" above):
export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
```

Plus [`uv`](https://docs.astral.sh/uv/). See `.env.example` for all the
variables.

## Run

```sh
cd examples/python/08-plan-execute
uv run main.py
uv run main.py --prompt "Compare three Rust web frameworks (axum, actix-web, rocket) on performance, ergonomics, and ecosystem; cite sources and save to async-comparison.md."
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint,
and `--model` overrides the id for a single run — same as 04 / 05 / 06.

`async-comparison.md` is written into `workspace/` and is `.gitignore`d so re-runs
stay clean.
```
