# 08 — Plan-and-execute (multi-step goal decomposition)

The first example to swap the **loop strategy**. The agent is given a multi-step
goal — _research three Rust async runtimes and write a cited comparison to a
file_ — and instead of reacting one tool call at a time, it **decomposes the goal
into a plan first**, prints that plan, then executes each subtask in turn.

The thesis: **swapping the loop strategy is a one-line harness change.** Same
builder, same sandbox, same tools as [`06`](../06-web-research). Only the loop
strategy on the `Task` changes — from `re_act` to `plan_execute` — and the
behavior changes qualitatively: the agent reasons about its own task structure
before touching any tool.

## The contrast with 06

|          | 06 — web-research                                    | 08 — plan-execute                                             |
| -------- | ---------------------------------------------------- | ------------------------------------------------------------- |
| Builder  | `conversational(model)`                              | `conversational(model)` _(same)_                              |
| Sandbox  | `WorkspaceScopedSandbox` over `workspace/`           | `WorkspaceScopedSandbox` over `workspace/` _(same)_           |
| Tools    | `web_search` (GET/JSON) + `write_file` + `read_file` | `web_search` (GET/JSON) + `write_file` + `read_file` _(same)_ |
| Strategy | `{ kind: "re_act", max_iterations: 10 }`             | **`{ kind: "plan_execute" }`** _(the swap)_                   |
| Behavior | reacts step-by-step, no upfront plan                 | **prints a full plan first, then runs each subtask**          |
| Output   | stream-printed `think` / `act` / `obs`               | a `── plan ──` banner + `[i/N]` subtask lines (via hooks)     |
| Side eff | writes `answer.md`                                   | writes `async-comparison.md`                                  |

Everything above the strategy row is held constant on purpose. The single
substantive change is the loop strategy:

```ts
// 06 — react step-by-step:
const task = newTask(prompt, SessionId.generate(), {
  kind: "re_act",
  max_iterations: 10,
});

// 08 — decompose the goal first, then execute each subtask:
const task = newTask(
  prompt,
  SessionId.generate(),
  { kind: "plan_execute" },
  { max_turns: 64 },
);
```

With `plan_execute`, the harness runs one constrained **planner turn** first. The
planner must return strict JSON `{ "tasks": [...], "rationale": ... }`; that text
is captured into a `PlanArtifact`, the plan is surfaced, and then each subtask
runs in its own bounded sub-loop. The turn budget is **divided across subtasks**
(per-task cap = remaining_turns / remaining_tasks), so we set a generous
`max_turns`.

## Watching the plan — via lifecycle hooks, not stream events

There are **no plan/subtask stream events**. The plan becomes visible through the
hook chain. This example registers a `PlanExecuteReporter` (`implements Hook`) on
two events:

- **`on_plan_created`** (`ctx.plan: PlanArtifact`) — fires once, after the plan
  is captured and before any subtask runs. We print the rationale and numbered
  tasks.
- **`on_task_advance`** (`ctx.task: Task`, `ctx.task_index`, `ctx.total_tasks`) —
  fires before each subtask. We print `[i/N] <instruction>` with
  `i = task_index + 1`; the subtask instruction is `ctx.task.instruction`.

```ts
const chain = new hooks.StandardHookChain();
chain.register(new PlanExecuteReporter());
const harness = HarnessBuilder.conversational(model)
  /* same sandbox + tools as 06 */
  .hooks(chain)
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

plan had 4 subtask(s)

async-comparison.md now exists on disk: …/workspace/async-comparison.md
```

In the TypeScript port the captured plan is persisted to the harness **RunStore**
seam (core #76), not to `session_state.extras`. The hook is therefore the
portable view: this example records the subtask count when `on_plan_created`
fires and prints it on success.

### The agent writes the file, not `main`

Just like 04 and 06, the file write happens **inside the loop** via the catalogue
`write_file` tool, backed by the `WorkspaceScopedSandbox`. `main.ts` never writes
`async-comparison.md` itself — the agent does, and the sandbox keeps it inside
`workspace/`.

## The search backend (SearXNG, GET + JSON)

There is **no live web-search backend in spore-core**. The endpoint is injected,
so you must supply one. The example reads it from `SPORE_WEB_SEARCH_ENDPOINT` and
exits if it is unset. `web_search` issues `GET <endpoint>?q=<query>` and hands the
response body back to the agent verbatim — identical wiring to
[`06`](../06-web-research).

This example targets a self-hosted **SearXNG** JSON API. The endpoint already
carries `format=json`; the GET path **preserves** that query string and appends
the `q` param, so the backend receives `GET /search?format=json&q=<query>`.

`web_search` is configured via `WebSearchTool.withConfig` (core
[issue #108](https://github.com/squirrelsoft-dev/spore-core/issues/108), now
implemented) with `method: "GET"` and `queryParam: "q"`. Auth-bearing backends
(Brave's `X-Subscription-Token`, Tavily's in-body `api_key`) are also supported
through the `authHeaders` / `bodyAuthParams` config fields; this example uses no
auth.

### SearXNG setup

1. Enable the JSON output format in your SearXNG `settings.yml`:

   ```yaml
   search:
     formats:
       - html
       - json
   ```

2. Restart SearXNG so the new format takes effect.

3. Point the example at it:

   ```sh
   export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
   ```

## An honesty note about PlanExecute

`plan_execute` is **largely untested against live models** and has real rough
edges you should know about before running:

- The planner must return **strict JSON** `{ "tasks": [...], "rationale": ... }`
  (a single ` ``` ` / ` ```json ` fence is tolerated and stripped). Anything else
  fails the run with `UnparseablePlan`.
- An **empty `tasks` array** halts with `HaltEmptyPlan`.
- A **subtask error aborts the whole run** with `HaltStepFailed` — there is no
  per-step retry.
- The turn **budget is divided across subtasks**, so a stingy `max_turns` starves
  later steps. Set it generously (this example uses `64`).
- **Small local models (e.g. `llama3.2`) often garble the plan JSON.** A larger
  hosted model produces a cleaner, more reliable demo. The harness is
  model-agnostic — swap the model interface and change nothing else. This example
  enables `structured_tool_calls` via
  `HarnessBuilder.modelParams({ structured_tool_calls: true, stop_sequences: [] })`
  to push small models toward one clean, schema-constrained tool call per turn
  across both the plan and execute phases.
- Subtask **inner tool calls stream in the TypeScript port** — unlike the
  Rust / Python / Go harnesses, which suppress the sub-loop stream. So in TS the
  `think` / `act` / `obs` lines also show each subtask's tool activity; the
  `on_plan_created` / `on_task_advance` hooks remain the portable, cross-language
  view of the plan and subtask boundaries.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
pnpm install
# A SearXNG JSON API (see "SearXNG setup" above):
export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
```

See `.env.example` for all the variables.

## Run

```sh
cd examples/typescript/08-plan-execute
pnpm start
pnpm start -- --prompt "Compare three Rust web frameworks (axum, actix-web, rocket) on performance, ergonomics, and ecosystem; cite sources and save to async-comparison.md."
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint,
and `--model` overrides the id for a single run — same as 04 / 05 / 06.

`async-comparison.md` is written into `workspace/` and is `.gitignore`d so re-runs
stay clean.
