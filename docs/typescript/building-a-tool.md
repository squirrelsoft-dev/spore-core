# Building a tool — TypeScript

> TypeScript code for the [building-a-tool guide](../guides/building-a-tool.md). Full runnable program:
> [`examples/typescript/03-tool-use`](../../examples/typescript/03-tool-use).

## Path 1 — catalogue tools

Pick a ready-made set and hand it to the builder. File tools need a real sandbox:

```ts
import {
  HarnessBuilder,
  OllamaModelInterface,
  WorkspaceScopedSandbox,
} from "@spore/core";
import { StandardTools } from "@spore/tools";

const workspace = new WorkspaceScopedSandbox({ root: process.cwd() }); // a sandbox rooted at your workspace
const harness = HarnessBuilder.conversational(
  new OllamaModelInterface("llama3.2"),
)
  .sandbox(workspace)
  .tools(StandardTools.codingSet())
  .build();
```

`.tool(t)` adds one tool; `.tools(iter)` adds many. The default `NullSandbox` denies environment
access, so swapping in `WorkspaceScopedSandbox` is what makes the file tools live.

## Path 3 — implement the registry directly

For pure-compute tools, implement the harness-loop `ToolRegistry` interface. `schemas()` is what
the model sees; `dispatch()` runs the call. No sandbox required.

```ts
import {
  HarnessBuilder,
  OllamaModelInterface,
  toolOutput,
  type ToolCall,
  type ToolOutput,
  type ToolRegistry,
  type ToolSchema,
} from "@spore/core";

class LocalTools implements ToolRegistry {
  schemas(): ToolSchema[] {
    return [
      {
        name: "calculator",
        description: "Compute a binary arithmetic operation.",
        parameters: {
          type: "object",
          properties: {
            a: { type: "number" },
            b: { type: "number" },
            op: { type: "string", enum: ["+", "-", "*", "/"] },
          },
          required: ["a", "b", "op"],
        },
        annotations: { read_only: true, destructive: false, idempotent: false, open_world: false },
      },
    ];
  }

  async dispatch(call: ToolCall): Promise<ToolOutput> {
    if (call.name !== "calculator") {
      return { kind: "error", message: `unknown tool ${call.name}`, recoverable: true };
    }
    const { a, b, op } = call.input as { a: number; b: number; op: string };
    switch (op) {
      case "+": return toolOutput.success(`${a + b}`);
      case "-": return toolOutput.success(`${a - b}`);
      case "*": return toolOutput.success(`${a * b}`);
      case "/": return toolOutput.success(`${a / b}`);
      default: return { kind: "error", message: `bad op ${op}`, recoverable: true };
    }
  }

  isAlwaysHalt(): boolean {
    return false;
  }

  hasSubagentTools(): boolean {
    return false;
  }
}

const harness = HarnessBuilder.conversational(
  new OllamaModelInterface("llama3.2"),
)
  .toolRegistry(new LocalTools())
  .build();
```

## Path 5 — a custom tool with the storage seam

A tool that persists durable state uses the `ToolContext`'s `runStore` (keyed by the run's
`SessionId` under a stable key). Register it through `StandardToolRegistry` so the harness bridges
the context into every dispatch:

```ts
import {
  HarnessBuilder,
  OllamaModelInterface,
  toolRegistry,
  toolOutput,
  type ToolCall,
  type ToolOutput,
} from "@spore/core";

const { StandardToolRegistry, ToolContext } = toolRegistry;

const registry = new StandardToolRegistry();
registry.register(
  {
    name: "remember",
    async execute(call: ToolCall, _sandbox, ctx): Promise<ToolOutput> {
      const { value } = call.input as { value: string };
      await ctx.runStore.put(ctx.sessionId, "memory", { value });
      return toolOutput.success(`remembered ${value}`);
    },
  },
  {
    name: "remember",
    description: "Persist a value to the run store.",
    parameters: { type: "object", properties: { value: { type: "string" } }, required: ["value"] },
    annotations: { read_only: false, destructive: false, idempotent: false, open_world: false },
  },
);

const harness = HarnessBuilder.conversational(
  new OllamaModelInterface("llama3.2"),
)
  .tools(registry.standardTools())
  .build();
```

## Run it

```sh
cd examples/typescript/03-tool-use
pnpm install
pnpm start
```