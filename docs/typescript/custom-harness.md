# Custom harness — TypeScript

> TypeScript code for the [custom-harness guide](../guides/custom-harness.md). Every method:
> [harness-builder reference](../reference/harness-builder.md).

## From the preset — override one slot at a time

`conversational(model)` returns a `HarnessBuilder` with every required component defaulted. Each
builder method replaces exactly one slot and returns the builder, so overrides chain:

```ts
import {
  HarnessBuilder,
  InMemoryStorageProvider,
  OllamaModelInterface,
  StorageProvider,
  WorkspaceScopedSandbox,
} from "@spore/core";
import { StandardTools } from "@spore/tools";

const workspace = new WorkspaceScopedSandbox({ root: process.cwd() });
const harness = HarnessBuilder.conversational(
  new OllamaModelInterface("llama3.2"),
)
  .sandbox(workspace)                              // let file tools touch a real dir
  .tools(StandardTools.codingSet())                 // catalogue tools
  .storage(StorageProvider.single(new InMemoryStorageProvider()))
  .autoPersistSessions(true)                        // load/save session state by id
  .systemPrompt("You are a careful coding assistant.")
  .build();
```

Anything you don't set keeps the preset default (`ModelAgent`, `EmptyToolRegistry`,
`NullSandbox`, `StandardContextManager`, `CompleteOnFinalResponse`).

## From scratch — the five required components

When no preset fits, call the `HarnessBuilder` constructor with the five required components
yourself; optional ones still default until set:

```ts
import { HarnessBuilder } from "@spore/core";

const harness = new HarnessBuilder(
  agent,               // Agent
  toolRegistry,        // ToolRegistry
  sandbox,             // SandboxProvider
  contextManager,      // ContextManager
  terminationPolicy,   // TerminationPolicy
)
  .observability(observability)   // optional, defaults to none
  .middleware(middleware)         // optional
  .build();
```

## Commonly overridden slots

| Method | Replaces | Default |
|--------|----------|---------|
| `.tool(t)` / `.tools(iter)` | adds catalogue tools | none |
| `.toolRegistry(r)` | the harness-loop tool registry | `EmptyToolRegistry` |
| `.sandbox(s)` | the sandbox | `NullSandbox` |
| `.systemPrompt(s)` | the system prompt | none |
| `.storage(s)` | the storage provider | no-op |
| `.autoPersistSessions(b)` | load/save session state by id | `false` |
| `.middleware(m)` | the middleware chain | none |
| `.observability(o)` | the observability provider | none |
| `.pricing(table)` | cost in usage reporting | `PricingTable.DEFAULT` |
| `.steeringSource(fn)` | mid-run user-message injection | none |
| `.toolCallRepair(r)` / `.maxRepairAttempts(n)` | malformed-arg repair seam | none / `1` |
| `.toolCallNudge(b)` | behavioral tool-call nudge | `false` |
| `.planEmptyMaxRetries(n)` | plan-empty re-ask budget | `0` |

See the [reference](../reference/harness-builder.md) for the full list (verifier, planner /
evaluator agents, VCS provider, metric evaluator, compaction settings, hooks).

## Notes

- **`.build()`** finalizes into a `StandardHarness`. Use `.buildConfig()` instead if you want the
  `HarnessConfig` without constructing the harness.
- **Order doesn't matter** — each setter is an independent slot. Override the same slot twice and
  the last call wins.
- **Sandbox is the gate.** Catalogue file/exec tools are inert under the default `NullSandbox`;
  give them a `WorkspaceScopedSandbox` to operate on a real directory.