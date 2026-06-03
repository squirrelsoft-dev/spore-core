# 05 — Custom sandboxed tool (`defineTool`)

The first example that ships a tool **you** wrote. Examples 03 and 04 used the
two built-in tool paths; this one uses spore-core's ergonomic extension helper —
`defineTool` — to add two custom tools, `remember(key, value)` and `recall(key)`,
and lets an agent research a topic across turns and summarize from the facts it
recalled.

The thesis is the same as 04, taken one step further: **the harness doesn't
change — only what you register does.** Same `conversational(model)` builder,
same ReAct loop, same stream-printed `think` / `act` / `obs`. The only new thing
is the registration pattern below.

## The pattern: `defineTool({ input, execute })` → `.tool()`

`defineTool` is the TypeScript analogue of Rust's `tool!` macro. You give it a
single **Zod schema** for the input and it does two things with that one schema:

- **Advertises** the tool by deriving its JSON-Schema `parameters` from the Zod
  schema (via `zod-to-json-schema`).
- **Validates** the model's raw arguments against the *same* schema before your
  `execute` body runs.

Because one schema serves both jobs, the classic drift between a hand-written
`parameters` blob and a separate validation schema is **eliminated by
construction** — there is no second source of truth to fall out of sync.

### 1. Describe the input with Zod

The Zod schema is the single source of truth. `.describe(...)` strings flow
straight into the advertised JSON schema the model sees:

```ts
import { z } from "zod";

export const RememberInput = z.object({
  key: z.string().describe("Short, stable key to file the fact under."),
  value: z.string().describe("The fact to remember."),
});
```

### 2. `defineTool({ name, description, input, execute, annotations? })`

`execute` receives the **already-validated, fully-typed** input, plus the two
seams every tool gets:

```ts
import { toolRegistry, toolOutput } from "@spore/core";

export function rememberTool() {
  return toolRegistry.defineTool({
    name: "remember",
    description: "Store a fact under a short key so it can be recalled later.",
    input: RememberInput,
    // annotations omitted → all-false; remember MUTATES, so it is NOT read_only.
    execute: async (input, _sandbox, ctx) => {
      await ctx.runStore.put(ctx.sessionId, `fact:${input.key}`, input.value);
      return toolOutput.success(`remembered ${input.key}`);
    },
  });
}
```

`execute` receives two seams:

- **`sandbox`** — the only path to the environment (filesystem, network). These
  two tools never touch it, so they name it `_sandbox`.
- **`ctx: ToolContext`** — the only path to durable, per-run state. Keys are
  namespaced under `fact:` so they cannot collide with reserved catalogue keys
  (`todo`, `task`, `memory`).

`remember` mutates shared state, so it omits annotations (all default to
`false`). `recall` only reads, so it passes
`annotations: { read_only: true, idempotent: true }`.

If the model sends arguments that don't match the schema, `defineTool` returns a
**recoverable** error whose message contains `invalid parameters` — so a
configured tool-call-repair pass can coerce the arguments and re-dispatch rather
than halting the run. You never write that validation by hand.

`defineTool` returns a `StandardTool` (`{ implementation, schema }`), the same
bundle the catalogue uses, with `implementation.name` and `schema.name` always
in agreement.

### 3. `.tool(...)`

Register each bundle on the builder — once per custom tool. The builder wires the
sandbox and a per-run `ToolContext` in automatically:

```ts
const harness = HarnessBuilder.conversational(model)
  .tool(rememberTool())
  .tool(recallTool())
  .systemPrompt(SYSTEM_PROMPT)
  .build();
```

## The contrast with 04

|         | 04 — filesystem-agent                         | 05 — custom-sandboxed-tool                      |
| ------- | --------------------------------------------- | ----------------------------------------------- |
| Builder | `conversational(model)`                       | `conversational(model)` _(same)_               |
| Loop    | ReAct                                         | ReAct _(same)_                                  |
| Tools   | `.tools(StandardTools.codingSet())`           | `.tool(rememberTool()).tool(recallTool())` _(your own)_ |
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
