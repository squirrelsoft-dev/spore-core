# 08 — Plan-and-execute (multi-step goal decomposition)

The first example to swap the **loop strategy**. The agent is given a multi-step
goal — *research three Rust async runtimes and write a cited comparison to a
file* — and instead of reacting one tool call at a time, it **decomposes the goal
into a plan first**, prints that plan, then executes each subtask in turn.

The thesis: **swapping the loop strategy is a one-line harness change.** Same
builder, same sandbox, same tools as [`06`](../06-web-research). Only the
`LoopStrategy` on the `Task` changes — from `StrategyReAct` to
`StrategyPlanExecute` — and the behavior changes qualitatively: the agent reasons
about its own task structure before touching any tool.

## The contrast with 06

|          | 06 — web-research                                  | 08 — plan-execute                                              |
| -------- | -------------------------------------------------- | -------------------------------------------------------------- |
| Builder  | `ConversationalBuilder(model)`                     | `ConversationalBuilder(model)` *(same)*                        |
| Sandbox  | `WorkspaceScopedSandbox` over `workspace/`         | `WorkspaceScopedSandbox` over `workspace/` *(same)*            |
| Tools    | `NewWebSearchToolFromConfig` + `WriteFile` + `ReadFile` | `NewWebSearchToolFromConfig` + `WriteFile` + `ReadFile` *(same)* |
| Strategy | `StrategyReAct` (`MaxIterations: 10`)              | **`StrategyPlanExecute`** *(the swap)*                         |
| Behavior | reacts step-by-step, no upfront plan               | **prints a full plan first, then runs each subtask**           |
| Output   | stream-printed `think` / `act` / `obs`             | a `── plan ──` banner + `[i/N]` subtask lines (via hooks)      |
| Side eff | writes `answer.md`                                 | writes `async-comparison.md`                                   |

Everything above the strategy row is held constant on purpose. The single
substantive change is the loop strategy:

```go
// 06 — react step-by-step:
task := sporecore.NewTask(prompt, sporecore.NewSessionID(),
    sporecore.LoopStrategy{Kind: sporecore.StrategyReAct, MaxIterations: 10})

// 08 — decompose the goal first, then execute each subtask:
task := sporecore.NewTask(prompt, sporecore.NewSessionID(),
    sporecore.LoopStrategy{Kind: sporecore.StrategyPlanExecute}).
    WithBudget(sporecore.BudgetLimits{MaxTurns: &maxTurns}) // generous: 64
```

With `StrategyPlanExecute`, the harness runs one constrained **planner turn**
first. The planner must return strict JSON `{ "tasks": [...], "rationale": ... }`;
that text is captured into a `PlanArtifact`, the plan is surfaced, and then each
subtask runs in its own bounded sub-loop. The turn budget is **divided across
subtasks** (per-task cap = remaining_turns / remaining_tasks), so we set a
generous `MaxTurns`.

## Watching the plan — via lifecycle hooks, not stream events

There are **no plan/subtask stream events**. The plan becomes visible through the
hook chain. This example registers a `planExecuteReporter` (a `sporecore.Hook`)
on two events:

- **`HookEventOnPlanCreated`** — fires once, after the plan is captured and
  before any subtask runs. We print the rationale and numbered tasks
  (`HookContext.Plan`, a `*PlanArtifact`).
- **`HookEventOnTaskAdvance`** — fires before each subtask. We print
  `[i/N] <instruction>` using `HookContext.Task.Instruction`,
  `HookContext.TaskIndex` (0-based), and `HookContext.TotalTasks`.

### The Go hooks-wiring asymmetry

Unlike the Rust / TypeScript / Python builders, the Go observability
`HarnessBuilder` has **no `.Hooks()` setter**. The Go path is: build the config
with `BuildConfig()`, set `cfg.Hooks` to the chain that carries the reporter,
then `NewStandardHarness(cfg)`:

```go
cfg := observability.ConversationalBuilder(mi).
    Sandbox(sandbox).                                          // same sandbox as 06
    Tool(tools.StandardTool{Implementation: webSearch, Schema: webSearch.Schema()}). // same tool as 06 (SearXNG GET/JSON)
    Tool(tools.StandardTools{}.WriteFile()).
    Tool(tools.StandardTools{}.ReadFile()).
    SystemPrompt(systemPrompt).
    BuildConfig()

chain := sporecore.NewStandardHookChain()
_ = chain.Register(planExecuteReporter{})
cfg.Hooks = chain                       // ← the wiring point (no builder setter)
harness := sporecore.NewStandardHarness(cfg)
```

`NewStandardHarness` only defaults `Hooks` when nil and auto-registers the Ralph
stop hook onto whatever chain is set, so registering the reporter on our chain
first is all that is needed.

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

async-comparison.md now exists on disk: …/workspace/async-comparison.md
```

### The agent writes the file, not `main`

Just like 04 and 06, the file write happens **inside the loop** via the catalogue
`write_file` tool, backed by the `WorkspaceScopedSandbox`. `main.go` never writes
`async-comparison.md` itself — the agent does, and the sandbox keeps it inside
`workspace/`.

## The search backend

There is **no live web-search backend in spore-core**. The endpoint is injected,
so you must supply one. The example reads it from `SPORE_WEB_SEARCH_ENDPOINT` and
exits with a clear error (exit code 2) if it is unset. `web_search` issues
`GET <endpoint>?q=<query>` and hands the response body back to the agent verbatim.
The GET path **preserves any query string already on the endpoint**, so a SearXNG
`/search?format=json` URL becomes `GET /search?format=json&q=<query>`.

A self-hosted **SearXNG** JSON API is the recommended backend (a local mock that
answers the same `GET ...?q=<query>` shape also works).

> Custom auth (Brave's `X-Subscription-Token`, Tavily's in-body `api_key`) is now
> supported by `web_search` via `WebSearchConfig.AuthHeaders` /
> `BodyAuthParams` (core [issue #108](https://github.com/squirrelsoft-dev/spore-core/issues/108),
> resolved). This example targets SearXNG, which needs no auth.

### SearXNG setup

1. Enable the JSON output format in your SearXNG `settings.yml`:

   ```yaml
   search:
     formats:
       - html
       - json
   ```

2. Restart SearXNG so the new format takes effect.
3. Point the example at the JSON API:

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
- The turn **budget is divided across subtasks**, so a stingy `MaxTurns` starves
  later steps. Set it generously (this example uses `64`).
- **Small local models (e.g. `llama3.2`) often garble the plan JSON.** A larger
  hosted model produces a cleaner, more reliable demo. The harness is
  model-agnostic — swap the model interface and change nothing else. This
  example enables `structured_tool_calls` via
  `.WithModelParams(sporecore.ModelParams{StructuredToolCalls: true})` to push
  small models toward one clean, schema-constrained tool call per turn (no
  interleaved reasoning) across both the plan and execute phases.
- Subtask **inner tool calls stream in the TypeScript port only**; the Go
  harness (like Rust and Python) suppresses the sub-loop stream, so the
  `OnPlanCreated` / `OnTaskAdvance` hooks are the portable, cross-language view
  of execution.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
# A SearXNG JSON API endpoint (see "SearXNG setup" above):
export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
```

See `.env.example` for all the variables.

## Run

```sh
cd examples/go/08-plan-execute
go run .
go run . --prompt "Compare three Rust web frameworks (axum, actix-web, rocket) on performance, ergonomics, and ecosystem; cite sources and save to async-comparison.md."
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint,
and `--model` overrides the id for a single run — same as 04 / 05 / 06.

`async-comparison.md` is written into `workspace/` and is `.gitignore`d so re-runs
stay clean.
