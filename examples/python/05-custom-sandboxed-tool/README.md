# 05 — Custom sandboxed tool (`impl Tool`)

The first example that ships a tool **you** wrote. Examples 03 and 04 used the
two built-in tool paths; this one uses spore-core's public extension point — the
[`Tool`](../../../python/packages/spore_core/src/spore_core/tool_registry.py)
protocol — to add two custom tools, `remember(key, value)` and `recall(key)`, and
lets an agent research a topic across turns and summarize from the facts it
recalled.

The thesis is the same as 04, taken one step further: **the harness doesn't
change — only what you register does.** Same `conversational(model)` builder,
same `ReAct` loop, same stream-printed `think` / `act` / `obs`. The only new
thing is the three-step registration pattern below.

## The pattern: `Tool` impl → `StandardTool` → `.tool()`

### 1. Implement the `Tool` protocol

A tool is any object that satisfies the structural `Tool` protocol. The
substance is `name()` and `execute()` (plus two `False`-returning flag methods):

```python
async def execute(
    self,
    call: ToolCall,
    sandbox: SandboxProvider,   # the environment seam (unused here)
    ctx: ToolContext,           # the storage seam
) -> ToolOutput:
    ...
```

`execute` receives two seams:

- **`sandbox`** — the only path to the environment (filesystem, network). These
  two tools never touch it, so they ignore it.
- **`ctx: ToolContext`** — the only path to durable, per-run state. `remember`
  calls `ctx.run_store.put(ctx.session_id, "fact:{key}", …)`; `recall` calls
  `ctx.run_store.get(…)`. Keys are namespaced under `fact:` so they cannot
  collide with reserved catalogue keys (`todo`, `task`, `memory`).

`remember` mutates shared state, so it is **not** `read_only`. `recall` only
reads, so it is `read_only` + `idempotent`.

#### Two different "tool" interfaces — don't confuse them

|                                   | Where it lives | Used by                                       |
| --------------------------------- | -------------- | --------------------------------------------- |
| harness-loop `ToolRegistry` (03)  | `spore_core`   | hand-rolled `schemas()` + `dispatch()`        |
| `Tool` protocol (this example)    | `spore_core`   | one tool, with the sandbox + `ToolContext` seams |

03 implemented the slim harness-loop `ToolRegistry` itself and dispatched every
call by hand. Here you implement the richer per-tool `Tool` protocol once per
tool and let the builder do the dispatch.

### 2. `StandardTool(implementation, schema)`

Bundle the implementation with its registry-side schema so the two can never
drift apart. `tool.name()` MUST equal `schema.name`:

```python
StandardTool(RememberTool(), RememberTool.schema())
```

### 3. `.tool(...)`

Register each bundle on the builder — once per custom tool. The builder wires the
sandbox and a per-run `ToolContext` in automatically:

```python
harness = (
    HarnessBuilder.conversational(model)
    .tool(StandardTool(RememberTool(), RememberTool.schema()))
    .tool(StandardTool(RecallTool(), RecallTool.schema()))
    .system_prompt(SYSTEM_PROMPT)
    .build()
)
```

## The contrast with 04

|              | 04 — filesystem-agent                          | 05 — custom-sandboxed-tool                       |
| ------------ | ---------------------------------------------- | ------------------------------------------------ |
| Builder      | `conversational(model)`                        | `conversational(model)` *(same)*                 |
| Loop         | `ReAct`                                        | `ReAct` *(same)*                                 |
| Tools        | `.tools(StandardTools.coding_set())`           | `.tool(remember).tool(recall)` *(your own)*      |
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
    act    → remember({"key": "habitat", "value": "Lives in coastal ocean waters worldwide."})
    obs → remembered habitat
    act    → remember({"key": "diet", "value": "Eats crabs, shrimp, and small fish."})
    obs → remembered diet
    …
think  · turn 3
    act    → recall({"key": "habitat"})
    obs → Lives in coastal ocean waters worldwide.

summary (3 turn(s)): The common octopus lives in coastal waters, eats crabs and …
```

A `recall` for a key that was never stored is a **recoverable** error
(`obs(err)→ no fact stored under '…'`) — the agent can adapt rather than the run
halting.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
```

Plus [`uv`](https://docs.astral.sh/uv/).

## Run

```sh
cd examples/python/05-custom-sandboxed-tool
uv run main.py
uv run main.py --prompt "Research mycelium. Remember a few facts, then recall and summarize them."
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint,
and `--model` overrides the id for a single run — same as 03 / 04.

## Tests

```sh
cd examples/python/05-custom-sandboxed-tool
uv run pytest
```

`test_tools.py` covers every rule: `remember` stores under the `fact:` prefix,
`recall` returns the stored value, a `recall` miss is a recoverable error,
missing/wrong arguments are recoverable errors, and the annotations
(`recall` read-only + idempotent, `remember` neither) are correct.
