# 05 — Custom tool with the `tool!` macro

The first example that ships a tool **you** wrote. Examples 03 and 04 used the
two built-in tool paths; this one uses spore-core's ergonomic
[`tool!`](https://docs.rs/spore-core) macro to add two custom tools,
`remember(key, value)` and `recall(key)`, and lets an agent research a topic
across turns and summarize from the facts it recalled.

The thesis is the same as 04, taken one step further: **the harness doesn't
change — only what you register does.** Same `conversational(model)` builder,
same `ReAct` loop, same stream-printed `think` / `act` / `obs`. The only new
thing is the two-step pattern below.

## The pattern: `tool! { .. }` → `.tool()`

### 1. `tool! { .. }`

A `tool!` block defines a tool from four fields — `name`, `description`, a typed
`input`, and an async `execute` closure — plus an optional `annotations`. It
expands to a `Tool` impl bundled with its schema into a `StandardTool`:

```rust
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RememberInput {
    pub key: String,
    pub value: String,
}

pub fn remember_tool() -> StandardTool {
    tool! {
        name: "remember",
        description: "Store a fact under a short key so it can be recalled later.",
        input: RememberInput,
        execute: |input, _sandbox, ctx| async move {
            let store_key = format!("fact:{}", input.key);
            match ctx.run_store().put(ctx.session_id(), &store_key, json!(input.value)).await {
                Ok(()) => ToolOutput::success(format!("remembered {}", input.key)),
                Err(e) => ToolOutput::error(format!("could not persist: {e}")),
            }
        },
    }
}
```

The macro **derives the advertised JSON schema from the `input` type** (via
`schemars`), so the schema the model sees and the struct the tool deserializes
can never drift. Bad arguments become a *recoverable* error automatically, so a
configured `ToolCallRepair` can coerce and retry.

The `execute` closure receives two seams:

- **`sandbox`** — the only path to the environment (filesystem, network). These
  two tools never touch it, so they name it `_sandbox`.
- **`ctx: &ToolContext`** — the only path to durable, per-run state. `remember`
  calls `ctx.run_store().put(ctx.session_id(), "fact:{key}", …)`; `recall` calls
  `ctx.run_store().get(…)`. Keys are namespaced under `fact:` so they cannot
  collide with reserved catalogue keys (`todo`, `task`, `memory`).

`remember` mutates shared state, so it omits `annotations` (default: not
`read_only`). `recall` passes
`annotations: ToolAnnotations { read_only: true, idempotent: true, .. }` because
it only reads.

#### Two different "tool" traits — don't confuse them

| Trait                                | Where it lives        | Used by                                  |
| ------------------------------------ | --------------------- | ---------------------------------------- |
| harness-loop `ToolRegistry` (03)     | `spore_core` (loop)   | hand-rolled `schemas()` + `dispatch()`   |
| `Tool` (this example, via `tool!`)   | `spore_core`          | one tool, with the sandbox + `ToolContext` seams |

03 implemented the slim harness-loop `ToolRegistry` itself and dispatched every
call by hand. Here the `tool!` macro generates the richer per-tool `Tool` impl
for you and the builder does the dispatch.

### 2. `.tool(...)`

Register each tool on the builder — once per custom tool. The builder wires the
sandbox and a per-run `ToolContext` in automatically:

```rust
let harness = HarnessBuilder::conversational(model)
    .tool(remember_tool())
    .tool(recall_tool())
    .system_prompt(SYSTEM_PROMPT)
    .build();
```

## The contrast with 04

|              | 04 — filesystem-agent                          | 05 — custom-sandboxed-tool                       |
| ------------ | ---------------------------------------------- | ------------------------------------------------ |
| Builder      | `conversational(model)`                        | `conversational(model)` *(same)*                 |
| Loop         | `ReAct`                                        | `ReAct` *(same)*                                 |
| Tools        | `.tools(StandardTools::coding_set())`          | `.tool(remember).tool(recall)` *(your own)*      |
| Sandbox      | `WorkspaceScopedSandbox` over `sample-files/`  | none — tools ignore the sandbox                  |
| Storage      | n/a                                            | auto in-memory (free once `.tool()` is present)  |

Two builder differences from 04: there is no catalogue `.tools(...)`, and no
explicit `.sandbox(...)` / `.storage(...)`. `build()` defaults storage to an
in-memory provider whenever `.tool()` tools are present, so the run store works
for free.

## Watching the loop

Act / Observe lines come from harness **stream events** — the builder dispatches
your tools internally, just as it does the catalogue in 04:

```
think  · turn 1
    act    → remember({"key":"habitat","value":"Lives in coastal ocean waters worldwide."})
    obs → remembered habitat
    act    → remember({"key":"diet","value":"Eats crabs, shrimp, and small fish."})
    obs → remembered diet
    …
think  · turn 3
    act    → recall({"key":"habitat"})
    obs → Lives in coastal ocean waters worldwide.

summary (3 turn(s)): The common octopus lives in coastal waters, eats crabs and …
```

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
```

## Run

```sh
cd examples/rust/05-custom-sandboxed-tool
cargo run
cargo run -- --prompt "Research mycelium. Remember a few facts, then recall and summarize them."
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint
(see `.env.example`), and `--model` overrides the id for a single run — same as
03 / 04.

[`Tool`]: https://docs.rs/spore-core
