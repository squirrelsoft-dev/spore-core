# 05 — Custom sandboxed tool (`DefineTool`)

The first example that ships a tool **you** wrote. Examples 03 and 04 used the
two built-in tool paths; this one uses the ergonomic `tools.DefineTool` helper —
the Go analog of Rust's `tool!` macro — to add two custom tools,
`remember(key, value)` and `recall(key)`, and lets an agent research a topic
across turns and summarize from the facts it recalled.

The thesis is the same as 04, taken one step further: **the harness doesn't
change — only what you register does.** Same `ConversationalBuilder(model)`,
same `ReAct` loop, same stream-printed `think` / `act` / `obs`. The only new
thing is the two-step `DefineTool` → `.Tool()` pattern below.

## The pattern: `DefineTool(...)` → `.Tool()`

### 1. `tools.DefineTool[T](name, description, annotations, schema, execFn)`

`DefineTool` collapses the per-tool boilerplate — the empty struct, the
`Name` / `IsSubagentTool` / `MayProduceLargeOutput` / `Schema` / `Execute`
methods, the input-JSON unmarshal — into a single call. You supply a typed input
struct `T`, the explicit parameter schema, the annotations, and a typed exec
function; it hands back a ready-to-register `sporecore.StandardTool`:

```go
func RememberTool() sporecore.StandardTool {
    return coretools.DefineTool(
        "remember",
        "Store a fact under a short key …",
        sporecore.ToolAnnotations{}, // not ReadOnly — this mutates shared state
        rememberSchema,              // explicit JSON-Schema object
        func(
            ctx context.Context,
            in rememberInput,             // already unmarshaled + validated
            _ sporecore.SandboxProvider,  // the environment seam (unused here)
            toolCtx *sporecore.ToolContext, // the storage seam
        ) sporecore.ToolOutput {
            // ...
        },
    )
}
```

The exec function receives two seams:

- **`sandbox`** — the only path to the environment (filesystem, network). These
  two tools never touch it, so they ignore it (`_`).
- **`toolCtx *sporecore.ToolContext`** — the only path to durable, per-run state.
  `remember` calls `toolCtx.Put(ctx, "fact:{key}", …)`; `recall` calls
  `toolCtx.Get(ctx, "fact:{key}")`. The session id is carried inside the context,
  so you never thread it by hand. Keys are namespaced under `fact:` so they
  cannot collide with reserved catalogue keys (`todo`, `task_list`, `memory`).

`remember` mutates shared state, so its annotations are **not** `ReadOnly`.
`recall` only reads, so it is `ReadOnly` + `Idempotent`.

`DefineTool` unmarshals the model's raw arguments into `T` before your exec
function runs. A missing required field, a wrong-typed field, or malformed JSON
becomes a **recoverable** `invalid parameters` tool error — so a configured
`ToolCallRepair` can coerce the arguments and re-dispatch rather than halting the
run. Inside the exec function, a missing key or a store failure is likewise a
recoverable error via `sporecore.NewToolOutputError(...)` (a `ToolOutput` error
variant, not a Go `error`).

#### Explicit schema by design

Rust's `tool!` macro derives the JSON schema from the input type (via
`schemars`), so the advertised schema and the deserialization can never drift.
Go's `DefineTool` takes the **explicit-schema** variant: you pass the parameter
schema directly. This keeps the core module dependency-free (no jsonschema
library in `go.mod`). Reflection-based derivation from `T` is a possible later
opt-in. With the explicit variant you keep the struct and the schema in sync by
hand — small structs make this trivial, and the recoverable `invalid parameters`
error catches the cases that slip through.

### 2. `.Tool(...)`

Register each bundle on the builder — once per custom tool. The builder wires the
sandbox and a per-run `ToolContext` in automatically:

```go
harness := observability.ConversationalBuilder(mi).
    Tool(tools.RememberTool()).
    Tool(tools.RecallTool()).
    SystemPrompt(systemPrompt).
    Build()
```

## The contrast with 04

|         | 04 — filesystem-agent                          | 05 — custom-sandboxed-tool                       |
| ------- | ---------------------------------------------- | ------------------------------------------------ |
| Builder | `ConversationalBuilder(model)`                 | `ConversationalBuilder(model)` *(same)*          |
| Loop    | `StrategyReAct`                                | `StrategyReAct` *(same)*                         |
| Tools   | `.Tools(StandardTools{}.CodingSet()...)`       | `.Tool(RememberTool()).Tool(RecallTool())` *(your own, via `DefineTool`)* |
| Sandbox | `WorkspaceScopedSandbox` over `sample-files/`  | none — tools ignore the sandbox                  |
| Storage | n/a                                            | auto in-memory (free once `.Tool()` is present)  |

Two builder differences from 04: there is no catalogue `.Tools(...)`, and no
explicit `.Sandbox(...)` / `.Storage(...)`. `Build()` defaults the run store to
an in-memory provider whenever `.Tool()` tools are present, so the run store
works for free — no on-disk side effects.

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
cd examples/go/05-custom-sandboxed-tool
go run .
go run . --prompt "Research mycelium. Remember a few facts, then recall and summarize them."
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint,
and `--model` overrides the id for a single run — same as 03 / 04.
