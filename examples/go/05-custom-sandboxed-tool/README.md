# 05 — Custom sandboxed tool (`impl Tool`)

The first example that ships a tool **you** wrote. Examples 03 and 04 used the
two built-in tool paths; this one uses spore-core's public extension point — the
`sporecore.Tool` interface — to add two custom tools, `remember(key, value)` and
`recall(key)`, and lets an agent research a topic across turns and summarize from
the facts it recalled.

The thesis is the same as 04, taken one step further: **the harness doesn't
change — only what you register does.** Same `ConversationalBuilder(model)`,
same `ReAct` loop, same stream-printed `think` / `act` / `obs`. The only new
thing is the three-step registration pattern below.

## The pattern: `impl Tool` → `StandardTool{...}` → `.Tool()`

### 1. Implement `sporecore.Tool`

A tool is anything that implements `sporecore.Tool`. You write four small
methods; `Execute` is where the work happens:

```go
func (RememberTool) Execute(
    ctx context.Context,
    call sporecore.ToolCall,
    _ sporecore.SandboxProvider, // the environment seam (unused here)
    toolCtx *sporecore.ToolContext, // the storage seam
) sporecore.ToolOutput {
    // ...
}
```

`Execute` receives two seams:

- **`sandbox`** — the only path to the environment (filesystem, network). These
  two tools never touch it, so they ignore it (`_`).
- **`toolCtx *sporecore.ToolContext`** — the only path to durable, per-run state.
  `remember` calls `toolCtx.Put(ctx, "fact:{key}", …)`; `recall` calls
  `toolCtx.Get(ctx, "fact:{key}")`. The session id is carried inside the context,
  so you never thread it by hand. Keys are namespaced under `fact:` so they
  cannot collide with reserved catalogue keys (`todo`, `task_list`, `memory`).

`remember` mutates shared state, so its schema is **not** `ReadOnly`. `recall`
only reads, so it is `ReadOnly` + `Idempotent`.

A wrong/missing argument, a missing key, or a store failure is a **recoverable**
tool error: it returns `sporecore.NewToolOutputError(...)` (a `ToolOutput` error
variant, not a Go `error`), so the agent can adapt rather than halting the run.

#### Two different "tool" traits — don't confuse them

|                                          | Where it lives | Used by                                          |
| ---------------------------------------- | -------------- | ------------------------------------------------ |
| harness-loop `ToolRegistry` (03)         | `sporecore`    | hand-rolled `ActiveSchemas()` + `Dispatch()`     |
| `Tool` (this example)                    | `sporecore`    | one tool, with the sandbox + `ToolContext` seams |

03 hand-built a `*sporecore.StandardToolRegistry` and registered each `Tool` on
it directly. Here you implement the same per-tool `Tool` interface but let the
builder fold it into the registry and do the dispatch for you. (In Go the
canonical `StandardToolRegistry` *is* the harness-loop registry — there is no
separate slim trait as in Rust — so the contrast is in the *registration path*,
not in two distinct interfaces.)

### 2. `sporecore.StandardTool{Implementation: tool, Schema: schema}`

Bundle the implementation with its registry-side schema so the two can never
drift apart. `tool.Name()` MUST equal `schema.Name`:

```go
sporecore.StandardTool{Implementation: tools.RememberTool{}, Schema: tools.RememberTool{}.Schema()}
```

### 3. `.Tool(...)`

Register each bundle on the builder — once per custom tool. The builder wires the
sandbox and a per-run `ToolContext` in automatically:

```go
harness := observability.ConversationalBuilder(mi).
    Tool(sporecore.StandardTool{Implementation: tools.RememberTool{}, Schema: tools.RememberTool{}.Schema()}).
    Tool(sporecore.StandardTool{Implementation: tools.RecallTool{}, Schema: tools.RecallTool{}.Schema()}).
    SystemPrompt(systemPrompt).
    Build()
```

## The contrast with 04

|         | 04 — filesystem-agent                          | 05 — custom-sandboxed-tool                       |
| ------- | ---------------------------------------------- | ------------------------------------------------ |
| Builder | `ConversationalBuilder(model)`                 | `ConversationalBuilder(model)` *(same)*          |
| Loop    | `StrategyReAct`                                | `StrategyReAct` *(same)*                         |
| Tools   | `.Tools(StandardTools{}.CodingSet()...)`       | `.Tool(remember).Tool(recall)` *(your own)*      |
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
