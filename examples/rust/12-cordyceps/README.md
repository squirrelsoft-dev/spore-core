# Example 12 — cordyceps (a basic plan→execute coding agent)

A super-basic **plan→execute coding agent** in a REPL. It started as
[`04-filesystem-agent`](../04-filesystem-agent/README.md) and grows it three ways:

1. the workspace sandbox is **read-write**, rooted at the directory you launch
   from (override with `--workspace`), and the agent gets the full
   `StandardTools::coding_set()` (read / write / edit / list / grep / find +
   `bash`), so it can actually change code in your project;
2. the single `harness.run(...)` is wrapped in a **REPL** — build the harness
   once, then read a task per line, carrying the conversation forward across
   turns; and
3. each turn runs the **`PlanExecute`** strategy instead of a bare `ReAct` loop —
   a plan phase turns your prompt into a task list, then an execute phase works
   that list to completion.

Same `conversational(model)` builder and the same stream-printed `think` / `act`
/ `obs` trace — the strategy is what changed.

## The contrast with 04

|            | 04 — filesystem-agent                       | 12 — coding agent                                 |
| ---------- | ------------------------------------------- | ------------------------------------------------- |
| Builder    | `conversational(model)`                     | `conversational(model)` *(same)*                  |
| Loop       | `ReAct`                                      | **`PlanExecute`** (plan → execute, ReAct per task) |
| Tools      | `coding_set()`                               | `coding_set()` *(same)*                           |
| Sandbox    | `WorkspaceScopedSandbox` (read-only effect) | `WorkspaceScopedSandbox` over the launch dir (read-write) |
| Driver     | one `harness.run(...)`                       | a REPL: harness built once, conversation threaded |

```rust
let harness = HarnessBuilder::conversational(model)
    .sandbox(Arc::new(sandbox))                 // read-WRITE workspace
    .tools(StandardTools::coding_set())         // read/write/edit/list/grep/find/bash
    .system_prompt(SYSTEM_PROMPT)
    .registry_schema("plan", json!({}))         // plan slot's output schema (resolved, not enforced)
    .hooks(plan_announcer())                     // print the plan on OnPlanCreated
    .build();

let session_id = SessionId::generate();         // one conversation for the REPL
let mut history: Option<SessionState> = None;

while let Some(prompt) = read_prompt() {        // ← the REPL
    let task = Task::new(prompt, session_id.clone(), plan_execute_strategy());
    let mut opts = HarnessRunOptions::new(task).with_stream(...);
    if let Some(state) = history.take() {       // carry prior turns forward
        opts = opts.with_session_state(state);
    }
    if let RunResult::Success { session_state, .. } = harness.run(opts).await {
        history = Some(session_state);          // remember for the next turn
    }
}

// each turn's strategy: plan → execute
fn plan_execute_strategy() -> LoopStrategy {
    LoopStrategy::PlanExecute(PlanExecuteConfig {
        // plan: a ReAct sub-loop that emits a JSON {"tasks":[…]} plan. The plan
        // slot is STRUCTURED, so its leaf must declare an output schema or
        // startup validation rejects the run (hence the registry_schema above).
        plan: Box::new(LoopStrategy::ReAct(ReactConfig {
            budget: BudgetPolicy::PerLoop { value: 12 },
            behavior: BudgetExhaustedBehavior::Escalate,
            agent: AgentRef(String::new()),     // default agent
            toolset: ToolsetRef(String::new()), // default toolset (coding_set)
            output: Some(SchemaRef("plan".into())),
        })),
        // execute: each task runs its own ReAct loop, in dependency order.
        execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(25))),
        plan_model: None,
        behavior: BudgetExhaustedBehavior::Escalate,
    })
}
```

### Plan, then execute

`PlanExecute` is a two-phase combinator that wraps two child strategies:

- **Plan.** One ReAct sub-loop (≤ 12 steps here) that may look around the
  codebase with the read tools, then replies with a single JSON object —
  `{"tasks": [...], "rationale": "..."}`. The harness seeds the "respond with a
  JSON plan" directive itself; the example only supplies the slot. Because the
  `plan` slot is **structured** (it must yield a typed task graph), startup
  validation rejects a bare `ReAct` there unless its leaf declares an `output`
  schema — so we register an empty schema under `"plan"` and point the leaf at it.
  With `enforce_output_schemas` off (the default) that schema is only *resolved*,
  never delivered to or enforced on the model.
- **Execute.** The harness parses the plan into a **task list** and walks it,
  running the `execute` child — a ReAct loop (≤ 25 steps per task) — once per
  ready task, in dependency order, until every task is `Completed`.

The task list is **durable** and project-scoped (derived from the workspace
root). So this REPL keeps working a list until it's done: if a turn runs out of
budget partway through, a later turn finds the unfinished tasks and **resumes
them instead of re-planning**. Each new prompt that *does* re-plan appends its
tasks to the same conversation.

The plan is printed the moment it's captured, via an `OnPlanCreated` hook
(`plan_announcer()`), so you see the task list before the execute phase starts.

### State lives in BOTH the session and on disk

Two things carry across REPL turns:

- **The conversation.** Each turn runs on one stable `SessionId`, and we thread
  the prior turn's `SessionState` back in. `RunResult::Success` returns the full
  post-run history *losslessly* (issue #102 — user turns, assistant tool-call
  turns, tool results, the final answer), and `with_session_state(..)` feeds it
  into the next run, where your new prompt is appended on top. So the agent
  remembers what you both said earlier — ask it to "now add tests for that" and
  it knows what "that" is. The conversational `ContextManager` compacts the
  history at 80% of the context window — which this example sizes to the model's
  real window (256K for gemma4, so it compacts around 205K) rather than the
  harness's 8K `gemma` fallback, so the conversation isn't summarized away
  prematurely. Type `clear` to start fresh.
- **The workspace.** Files the agent wrote on an earlier turn are still on disk,
  so it can read, edit, and build on them — independently of the conversation.

### Watching the loop

```
code> create a hello.py that prints the first 10 fibonacci numbers, then run it
   think · turn 1
📋 plan (2 task(s)):
   1. Write hello.py with a loop that prints the first 10 fibonacci numbers
   2. Run hello.py and confirm the output
   think · turn 2
💬 Writing hello.py with a fibonacci loop.
   act → write_file({"path":"hello.py","content":"…"})
   obs → wrote hello.py
   think · turn 3
💬 Running it to check the output.
   act → bash({"command":"python3 hello.py"})
   obs → 0 1 1 2 3 5 8 13 21 34

answer (3 turn(s)): I created hello.py and ran it — it prints the first 10 …
```

The `📋 plan` block is printed by the `OnPlanCreated` hook the moment the plan
phase captures its JSON plan, before the execute phase runs the tasks.

The `💬` lines are the agent narrating itself through the `send_message` tool
(part of `coding_set()`). They're the **section headers** — printed in bright
white, flush left — while the `think` / `act` / `obs` mechanics are dimmed and
indented beneath them, so the trace reads as "what the agent is doing" with the
plumbing tucked underneath. The system prompt tells the agent the user only ever
sees these messages and the final answer — not its reasoning or raw tool calls —
so it emits a one-sentence `send_message` *in parallel* with the tool doing the
work, keeping you in the loop without spending an extra turn.

### Esc to abort (without losing context)

A run going sideways? Press **Esc** and it drops back to the `code>` prompt. The
run executes with the terminal in raw mode alongside a background key watcher
(`run_turn`); an Esc drops the `harness.run(..)` future, which cancels the
in-flight turn at its next await point.

The catch: a dropped future never hands back its `session_state`, so naively the
aborted turn's work would vanish and a follow-up `continue` would have nothing to
go on. To avoid that, the REPL **mirrors the turn from the stream** as it runs —
each event carries the `call_id` that pairs a tool result to its call — and on
abort splices that partial transcript (this turn's prompt + the tool calls and
results that completed) onto the prior history. A successful turn just uses the
harness's own lossless `session_state`; reconstruction is only the abort path. So
you can abort, type `continue`, and the agent still knows what it was doing.

(Esc-to-abort needs a TTY; piped/non-interactive stdin just runs without it.)

## Prerequisites

```sh
ollama serve &
ollama pull gemma4:e4b
```

Run **gemma4:e4b or better** — small models (e.g. llama3.2 3B) narrate tool use
instead of emitting tool calls, so they never act.

## Run

The agent's workspace root defaults to **the directory you launch from**, so run
it from the project you want it to work on:

```sh
# from the repo root — operates on the repo:
cargo run --manifest-path examples/rust/12-cordyceps/Cargo.toml -- --model gemma4:31b-cloud

# point it at an explicit directory instead:
cargo run --manifest-path examples/rust/12-cordyceps/Cargo.toml -- --workspace /path/to/project
```

Overrides:

- `--workspace <path>` / `SPORE_WORKSPACE` — the workspace root (default: cwd).
- `--model <id>` / `SPORE_OLLAMA_MODEL` — the model (default: `gemma4:e4b`).
- `--context-window <tokens>` / `SPORE_CONTEXT_WINDOW` — the model's **total**
  context window (default: `256000`, gemma4's real window). Compaction fires at
  80% of it (`should_compact`: `used / window >= threshold`), i.e. ~204,800
  tokens, leaving headroom for the turn that trips it. The harness's auto-resolver
  only knows `gemma → 8192`, which would compact ~30× too early, so this example
  sets the window explicitly. Lower it if you run a smaller-context model — the
  value is used as-is and is **not** clamped to the model's true window.
- `SPORE_OLLAMA_BASE_URL` — the Ollama endpoint.

> The sandbox is **read-write** and rooted at your workspace, so the agent can
> create, edit, and run files there. Launch it from a directory you're happy to
> let it modify (it's confined to that root — it can't escape it).
