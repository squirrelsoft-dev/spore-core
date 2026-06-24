# Example 12 — cordyceps (a basic plan→execute coding agent)

A super-basic **plan→execute coding agent** in a REPL. It started as
[`04-filesystem-agent`](../04-filesystem-agent/README.md) and grows it three ways:

1. it is built from the **`HarnessBuilder::coding_agent`** preset, which in one
   call wires the things a coding agent always needs: a **read-write** workspace
   sandbox rooted at the directory you launch from (override with `--workspace`),
   the full `StandardTools::coding_set()` (read / write / edit / list / grep /
   find + `bash`), a built-in coding system prompt, and `EscalationMode::AutoContinue`
   — so it can change code, and a spent step budget keeps working in-process;
2. the single `harness.run(...)` is wrapped in a **REPL** — build the harness
   once, then read a task per line, carrying the conversation forward across
   turns; and
3. each turn runs the **`PlanExecute`** strategy instead of a bare `ReAct` loop —
   a plan phase turns your prompt into a task list, then an execute phase works
   that list to completion.

The preset builds on the same `conversational(model)` core and the same
stream-printed `think` / `act` / `obs` trace — the strategy and the autonomous
escalation are what changed. The example then layers its own extras (skills, a
richer prompt, the plan-announcer hook) on top of the preset.

## The contrast with 04

|            | 04 — filesystem-agent                       | 12 — coding agent                                 |
| ---------- | ------------------------------------------- | ------------------------------------------------- |
| Builder    | `conversational(model)`                     | **`coding_agent(model, workspace)`** (preset)      |
| Loop       | `ReAct`                                      | **`PlanExecute`** (plan → execute, ReAct per task) |
| Tools      | `coding_set()` *(wired by hand)*             | `coding_set()` *(wired by the preset)*            |
| Sandbox    | `WorkspaceScopedSandbox` (read-only effect) | `WorkspaceScopedSandbox` over the launch dir (read-write, by the preset) |
| Budget     | escalates / surfaces                         | **`AutoContinue`** — grants in-process, no drive loop |
| Driver     | one `harness.run(...)`                       | a REPL: harness built once, conversation threaded |

```rust
// One call wires the coding-agent essentials — a read-WRITE workspace sandbox,
// coding_set() tools, a built-in coding system prompt, and AutoContinue (so a
// spent step budget keeps working in-process instead of pausing). SC-8.
let harness = HarnessBuilder::coding_agent(model, &workspace_root)?
    .system_prompt(SYSTEM_PROMPT)               // richer: adds skills + the plan-JSON exception
    .hooks(plan_announcer())                     // print the plan on OnPlanCreated
    // (this example also layers skills — a load_skill tool + a manifest-injecting
    //  context manager — see "Skills" below)
    .build();

let session_id = SessionId::generate();         // one conversation for the REPL
let mut history: Option<SessionState> = None;

while let Some(prompt) = read_prompt() {        // ← the REPL
    let task = Task::new(prompt, session_id.clone(), plan_execute_strategy());
    let mut opts = HarnessRunOptions::new(task);
    opts.on_stream = Some(sink.clone());
    if let Some(state) = &history {             // carry prior turns forward
        opts = opts.with_session_state(state.clone());
    }
    // One run carries the whole turn: AutoContinue grants more budget in-process,
    // so there is no consumer-side drive/resume loop. Esc-abortable throughout.
    if let Some(RunResult::Success { session_state, .. }) =
        run_abortable(harness.run(opts)).await
    {
        history = Some(session_state);          // remember for the next turn
    }
}

// each turn's strategy: plan → execute
fn plan_execute_strategy() -> LoopStrategy {
    LoopStrategy::PlanExecute(PlanExecuteConfig {
        // plan: a ReAct sub-loop that emits a JSON {"tasks":[…]} plan. SC-1 lets a
        // structured slot omit its output schema (absent ⇒ accept-all), so no
        // registry stamp is needed just to pass startup validation — the plan
        // phase's own "respond with a single JSON plan" directive drives the format.
        plan: Box::new(LoopStrategy::ReAct(ReactConfig {
            budget: BudgetPolicy::PerLoop { value: 12 },
            behavior: BudgetExhaustedBehavior::Escalate,
            agent: AgentRef(String::new()),     // default agent
            toolset: ToolsetRef(String::new()), // default toolset (coding_set)
            output: None,
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
  JSON plan" directive itself; the example only supplies the slot. The `plan` slot
  is **structured**, but SC-1 lets a structured slot omit its `output` schema (an
  absent schema is treated as accept-all), so the leaf carries `output: None` and
  no registry stamp is needed just to pass startup validation — the plan phase's
  own directive drives the format.
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

### Working the list to completion (AutoContinue — in the harness, not the REPL)

Every node runs under a finite step budget (the execute leaf is `PerLoop { 25 }`),
so a meaty task can spend it mid-flight. The `coding_agent` preset sets
`EscalationMode::AutoContinue` (SC-5), so when that happens the **harness** grants
more budget IN-PROCESS and keeps working — raising the exhausted scope's cap and
re-seeding the stalled worker, so the in-flight task picks up exactly where it
left off, no work lost. It does this up to `PRESET_MAX_AUTO_GRANTS` (10) grants of
`PRESET_STEPS_PER_GRANT` (25) steps each.

This is the part that used to be the consumer's job: earlier this example
hand-rolled a `drive()` loop that watched for `WaitingForHuman { BudgetExhausted }`
and resumed with a `ContinueWithBudget` grant. With the preset there is **no such
loop** — `harness.run(..)` returns a terminal result directly, and the same stream
sink narrates every in-process grant:

```
   act → write_file({"path":"greeting.txt", …})
   obs → wrote 13 bytes to greeting.txt
   think · turn 2
   act → read_file({"path":"greeting.txt"})
   obs → Hello, spore
answer (…): created greeting.txt and confirmed its contents.
```

The run is still Esc-abortable throughout (`run_abortable`). Past the grant cap
the run ends with `Failure`, but the plan isn't lost — it's durable, so a
follow-up prompt resumes the remaining tasks.

### State lives in BOTH the session and on disk

Two things carry across REPL turns:

- **The conversation.** Each turn runs on one stable `SessionId`, and we thread
  the prior turn's `SessionState` back in. `RunResult::Success` returns the full
  post-run history *losslessly* (issue #102 — user turns, assistant tool-call
  turns, tool results, the final answer), and `with_session_state(..)` feeds it
  into the next run, where your new prompt is appended on top. So the agent
  remembers what you both said earlier — ask it to "now add tests for that" and
  it knows what "that" is. The conversational `ContextManager` compacts the
  history at 80% of the context window — which this example sizes with ONE call,
  `OllamaModelInterface::with_context_window` (SC-4): it sets Ollama's `num_ctx`
  AND the window `provider()` reports, and the compaction budget auto-derives from
  that provider window (so there's no separate `context_length` to keep in sync).
  Sized to the model's real window (256K for gemma4, so it compacts around 205K)
  rather than the harness's 8K `gemma` fallback, the conversation isn't summarized
  away prematurely. Type `clear` to start fresh.
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
(`run_abortable`); an Esc drops the `harness.run(..)` future, which cancels the
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

## Skills (progressive disclosure)

This agent supports [**Agent Skills**](https://agentskills.io/specification) —
reusable, named procedures the agent pulls into context only when a task calls for
them. A skill is a directory with a `SKILL.md`: YAML frontmatter (`name` +
`description`) followed by a markdown procedure body.

Skills are discovered at startup from, in precedence order (last wins):

1. `skills/<name>/SKILL.md` next to this example (bundled — ships with it);
2. `.spore/skills/<name>/SKILL.md` under the agent's workspace;
3. `~/.spore/skills/<name>/SKILL.md` (your user skills).

### Two tiers, loaded on demand

Following the spec's **progressive disclosure**, the agent never holds every skill
body in context. It sees only a cheap **manifest** every turn — each skill's
`name` + one-line `description` — and pulls in a full body only when it decides the
skill is relevant:

- **Manifest (always).** `SkillInjectingContextManager` prepends `AVAILABLE
  SKILLS:` + one `name: description` line per skill to every turn (~tier 1).
- **Body (on load, then sticky).** A skill's full `SKILL.md` body is injected only
  once it is **active**, and every turn thereafter (tier 2).

### Three ways a skill activates

- **The agent loads it** via a `load_skill(name)` tool. The spec puts trigger
  keywords *inside* the `description` ("Use when …"), so when your request matches
  a skill's description the agent calls `load_skill` first, then follows the
  procedure. (The advanced case is the same tool driven by the agent's own
  judgement over the descriptions — no literal keyword needed.)
- **You load it** — type `/<name>` in the REPL (e.g. `/security-review`) to flip a
  skill active yourself. `/<name> <task>` loads it and runs `<task>` in one line.
  `/skills` lists what's available and what's active.
- (`clear` resets the conversation **and** the active-skill set.)

```
code> create a file named greeting.txt that greets the project
   think · turn 1
📋 plan (2 task(s)):
   1. load_skill('greeting-protocol')
   2. write_file('greeting.txt', …)
   act → load_skill({"name":"greeting-protocol"})
   obs → Loaded skill 'greeting-protocol' — its full procedure is now in your context. Follow it.
   act → write_file({"path":"greeting.txt","content":"GREETING-PROTOCOL-V1\n…"})
   obs → wrote 46 bytes to greeting.txt
answer: created greeting.txt following the active protocol.
```

The prompt's "greeting" matched the skill's `description`, so the agent loaded it
on its own and the injected body shaped what it wrote (the mandated first line).

### Why it's wired in the example, not the harness

Issue #9 added the `skill` guide type and the rich context manager can inject
skills structurally — but the **live** harness loop assembles context through the
pass-through compaction adapter, not the rich `assemble` (Known Deviation #8 /
issue #115). So skills reach the model only if the example injects them. We do that
by **wrapping** the compaction adapter: `SkillInjectingContextManager` forwards
every seam method (compaction included) to the inner adapter and only *prepends*
the manifest + active bodies in `assemble`. Issue #115 will fold discovery +
`load_skill` + sticky injection into the harness itself.

> No skills ship in this example by default — drop a `SKILL.md` under
> `skills/<name>/` (see [`skills/README.md`](skills/README.md)) and restart to see
> it in the manifest.

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
