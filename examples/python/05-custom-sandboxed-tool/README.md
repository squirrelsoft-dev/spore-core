# 05 — Custom sandboxed tool (`define_tool`)

The first example that ships a tool **you** wrote. Examples 03 and 04 used the
two built-in tool paths; this one uses spore-core's ergonomic
[`define_tool`](../../../python/packages/spore_tools/src/spore_tools/tools/define.py)
helper to add two custom tools, `remember(key, value)` and `recall(key)`, and
lets an agent research a topic across turns and summarize from the facts it
recalled.

The thesis is the same as 04, taken one step further: **the harness doesn't
change — only what you register does.** Same `conversational(model)` builder,
same `ReAct` loop, same stream-printed `think` / `act` / `obs`. The only new
thing is the registration pattern below.

## The headline property: the input model is the single source of truth

`define_tool` takes a typed [pydantic](https://docs.pydantic.dev) input model and
**derives the advertised JSON schema from it** (via `model_json_schema()`) — the
schema the model sees is never hand-written, so it can never drift from the model
the tool actually validates. This mirrors Rust's `tool!` macro (which derives the
schema from the input struct via `schemars`).

## The pattern: `define_tool(...)` → `.tool()`

### 1. Define a typed input model + an async `execute` body

```python
from pydantic import BaseModel

class RememberInput(BaseModel):
    key: str
    value: str

async def _remember(input: RememberInput, sandbox, ctx) -> ToolOutput:
    ...
```

`execute` receives the **validated** input model plus two seams:

- **`sandbox`** — the only path to the environment (filesystem, network). These
  two tools never touch it, so they ignore it.
- **`ctx: ToolContext`** — the only path to durable, per-run state. `remember`
  calls `ctx.run_store.put(ctx.session_id, "fact:{key}", …)`; `recall` calls
  `ctx.run_store.get(…)`. Keys are namespaced under `fact:` so they cannot
  collide with reserved catalogue keys (`todo`, `task`, `memory`).

`remember` mutates shared state, so it is **not** `read_only`. `recall` only
reads, so it is `read_only` + `idempotent` (passed via `annotations`).

### 2. `define_tool(...)` → `StandardTool`

`define_tool` returns a `StandardTool` bundling a `Tool` implementation with its
*derived* schema, so the schema and the validation can never drift:

```python
from spore_tools import define_tool

def remember_tool() -> StandardTool:
    return define_tool(
        name="remember",
        description="Store a fact under a short key …",
        input_model=RememberInput,
        execute=_remember,
        # annotations omitted → all-False (remember mutates state)
    )
```

If the model sends bad arguments (missing field, wrong type), the tool returns a
**recoverable** `invalid parameters for tool ...` error so a tool-call-repair pass
can coerce the arguments and retry — rather than halting the run.

### 3. `.tool(...)`

Register each tool on the builder — once per custom tool. The builder wires the
sandbox and a per-run `ToolContext` in automatically:

```python
harness = (
    HarnessBuilder.conversational(model)
    .tool(remember_tool())
    .tool(recall_tool())
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
missing/wrong arguments are recoverable `invalid parameters` errors (so tool-call
repair can retry), a store failure is recoverable (via a `FailingRunStore`), and
the annotations (`recall` read-only + idempotent, `remember` neither) and the
derived schema are correct.
