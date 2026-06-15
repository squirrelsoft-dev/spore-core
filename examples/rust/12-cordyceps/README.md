# Example 12 — cordyceps (a basic ReAct coding agent)

A super-basic **ReAct coding agent** in a REPL. It is
[`04-filesystem-agent`](../04-filesystem-agent/README.md) with two changes:

1. the workspace sandbox is **read-write** and the agent gets the full
   `StandardTools::coding_set()` (read / write / edit / list / grep / find +
   `bash`), so it can actually change code; and
2. the single `harness.run(...)` is wrapped in a **REPL** — build the harness
   once, then read a task per line and run a fresh ReAct loop for each.

Everything else is 04 verbatim: the same `conversational(model)` builder, the
same `ReAct` loop, the same stream-printed `think` / `act` / `obs` trace.

## The contrast with 04

|            | 04 — filesystem-agent                       | 12 — coding agent                                 |
| ---------- | ------------------------------------------- | ------------------------------------------------- |
| Builder    | `conversational(model)`                     | `conversational(model)` *(same)*                  |
| Loop       | `ReAct`                                      | `ReAct` *(same)*                                  |
| Tools      | `coding_set()`                               | `coding_set()` *(same)*                           |
| Sandbox    | `WorkspaceScopedSandbox` (read-only effect) | `WorkspaceScopedSandbox` over `workspace/` (read-write) |
| Driver     | one `harness.run(...)`                       | a REPL: harness built once, run per typed task    |

```rust
let harness = HarnessBuilder::conversational(model)
    .sandbox(Arc::new(sandbox))                 // read-WRITE workspace/
    .tools(StandardTools::coding_set())         // read/write/edit/list/grep/find/bash
    .system_prompt(SYSTEM_PROMPT)
    .build();

while let Some(prompt) = read_prompt() {        // ← the REPL
    let task = Task::new(prompt, SessionId::generate(),
        LoopStrategy::ReAct(ReactConfig::per_loop(25)));
    harness.run(HarnessRunOptions::new(task).with_stream(...)).await;
}
```

### State lives on disk, not in the session

Each REPL turn is a **fresh** `Task` with a fresh `SessionId` — the conversation
starts clean every time. What carries across turns is the **workspace**: files
the agent wrote on an earlier turn are still there, so a later turn can read,
edit, and build on them. (Threading the `SessionState` across turns to keep the
conversation itself would be a natural next step — this version keeps it simple.)

### Watching the loop

```
code> create a hello.py that prints the first 10 fibonacci numbers, then run it
think  · turn 1
    act    → write_file({"path":"hello.py","content":"…"})
    obs → wrote hello.py
    act    → bash({"command":"python3 hello.py"})
    obs → 0 1 1 2 3 5 8 13 21 34
think  · turn 2

answer (2 turn(s)): I created hello.py and ran it — it prints the first 10 …
```

## Prerequisites

```sh
ollama serve &
ollama pull gemma4:e4b
```

Run **gemma4:e4b or better** — small models (e.g. llama3.2 3B) narrate tool use
instead of emitting tool calls, so they never act.

## Run

```sh
cd examples/rust/12-cordyceps
cargo run                          # type a coding task; Ctrl-D to quit
cargo run -- --model qwen2.5-coder # override the model for one run
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint
(see `.env.example`); `--model` overrides the id for a single run.

The agent works inside `workspace/` (created on first run). Its contents are
`.gitignore`d so re-runs stay clean.
