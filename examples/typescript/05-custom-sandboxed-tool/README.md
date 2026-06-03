# 05 — Custom sandboxed tool (`impl Tool`)

The first example that ships a tool **you** wrote. Examples 03 and 04 used the
two built-in tool paths; this one uses spore-core's public extension point — the
`Tool` interface — to add two custom tools, `remember(key, value)` and
`recall(key)`, and lets an agent research a topic across turns and summarize from
the facts it recalled.

The thesis is the same as 04, taken one step further: **the harness doesn't
change — only what you register does.** Same `conversational(model)` builder,
same ReAct loop, same stream-printed `think` / `act` / `obs`. The only new thing
is the three-step registration pattern below.

## The pattern: implement `Tool` → wrap in a `StandardTool` → `.tool()`

### 1. Implement `Tool`

A tool is a class implementing `Tool` — you write `name` and `execute()`:

```ts
async execute(
  call: ToolCall,
  _sandbox: SandboxProvider, // the environment seam (unused here)
  ctx: ToolContext, // the storage seam
): Promise<ToolOutput> {
  /* ... */
}
```

`execute` receives two seams:

- **`sandbox`** — the only path to the environment (filesystem, network). These
  two tools never touch it, so they name it `_sandbox`.
- **`ctx: ToolContext`** — the only path to durable, per-run state. `remember`
  calls `ctx.runStore.put(ctx.sessionId, "fact:{key}", …)`; `recall` calls
  `ctx.runStore.get(…)`. Keys are namespaced under `fact:` so they cannot
  collide with reserved catalogue keys (`todo`, `task`, `memory`).

`remember` mutates shared state, so it is **not** `read_only`. `recall` only
reads, so it is `read_only` + `idempotent`.

#### Two different "tool" surfaces — don't confuse them

| Surface                          | Where it lives | Used by                                            |
| -------------------------------- | -------------- | -------------------------------------------------- |
| harness-loop `ToolRegistry` (03) | `@spore/core`  | hand-rolled `schemas()` + `dispatch()`             |
| `Tool` interface (this example)  | `@spore/core`  | one tool, with the sandbox + `ToolContext` seams   |

03 implemented the slim harness-loop `ToolRegistry` itself and dispatched every
call by hand. Here you implement the richer per-tool `Tool` interface once per
tool and let the builder do the dispatch.

### 2. Wrap in a `StandardTool`

Bundle the implementation with its registry-side schema so the two can never
drift apart. The bundle is a plain `{ implementation, schema }` object;
`tool.name` MUST equal `schema.name`:

```ts
const remember: StandardTool = {
  implementation: new RememberTool(),
  schema: RememberTool.schema(),
};
```

### 3. `.tool(...)`

Register each bundle on the builder — once per custom tool. The builder wires the
sandbox and a per-run `ToolContext` in automatically:

```ts
const harness = HarnessBuilder.conversational(model)
  .tool(remember)
  .tool(recall)
  .systemPrompt(SYSTEM_PROMPT)
  .build();
```

## The contrast with 04

|         | 04 — filesystem-agent                         | 05 — custom-sandboxed-tool                      |
| ------- | --------------------------------------------- | ----------------------------------------------- |
| Builder | `conversational(model)`                       | `conversational(model)` _(same)_               |
| Loop    | ReAct                                         | ReAct _(same)_                                  |
| Tools   | `.tools(StandardTools.codingSet())`           | `.tool(remember).tool(recall)` _(your own)_    |
| Sandbox | `WorkspaceScopedSandbox` over `sample-files/` | none — tools ignore the sandbox                 |
| Storage | n/a                                           | auto in-memory (free once `.tool()` is present) |

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
cd examples/typescript/05-custom-sandboxed-tool
pnpm install
pnpm start
pnpm start -- --prompt "Research mycelium. Remember a few facts, then recall and summarize them."
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint,
and `--model` overrides the id for a single run — same as 03 / 04.

## Tests

The two tools are unit-tested directly over an in-memory run store (no model
needed):

```sh
pnpm test
```
