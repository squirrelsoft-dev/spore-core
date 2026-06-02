# Custom harness — Rust

> Rust code for the [custom-harness guide](../guides/custom-harness.md). Every method:
> [harness-builder reference](../reference/harness-builder.md).

## From the preset — override one slot at a time

`conversational(model)` returns a `HarnessBuilder` with every required component defaulted. Each
builder method replaces exactly one slot and returns the builder, so overrides chain:

```rust
use std::sync::Arc;
use spore_core::{
    HarnessBuilder, OllamaModelInterface, StandardTools, StorageProvider, InMemoryStorageProvider,
};
use spore_core::sandbox::WorkspaceScopedSandbox;

let harness = HarnessBuilder::conversational(OllamaModelInterface::new("llama3.2"))
    .sandbox(workspace)                       // let file tools touch a real dir
    .tools(StandardTools::coding_set())       // catalogue tools
    .storage(Arc::new(StorageProvider::single(Arc::new(InMemoryStorageProvider::new()))))
    .auto_persist_sessions(true)              // load/save session state by id
    .system_prompt("You are a careful coding assistant.")
    .build();
```

Anything you don't set keeps the preset default (`ModelAgent`, `EmptyToolRegistry`,
`NullSandbox`, `StandardContextManager`, `CompleteOnFinalResponse`).

## From scratch — the five required components

When no preset fits, call `HarnessBuilder::new(...)` with the five required components yourself;
optional ones still default until set:

```rust
use std::sync::Arc;
use spore_core::HarnessBuilder;

let harness = HarnessBuilder::new(
    agent,               // Arc<dyn Agent>
    tool_registry,       // Arc<dyn ToolRegistry>
    sandbox,             // Arc<dyn SandboxProvider>
    context_manager,     // Arc<dyn ContextManager>
    termination_policy,  // Arc<dyn TerminationPolicy>
)
.observability(observability)   // optional, defaults to none
.middleware(middleware)         // optional
.build();
```

## Commonly overridden slots

| Method | Replaces | Default |
|--------|----------|---------|
| `.tool(t)` / `.tools(iter)` | adds catalogue tools | none |
| `.tool_registry(r)` | the harness-loop tool registry | `EmptyToolRegistry` |
| `.sandbox(s)` | the sandbox | `NullSandbox` |
| `.system_prompt(s)` | the system prompt | none |
| `.storage(s)` | the storage provider | `StorageProvider::no_op()` |
| `.auto_persist_sessions(b)` | load/save session state by id | `false` |
| `.middleware(m)` | the middleware chain | none |
| `.observability(o)` | the observability provider | none |
| `.with_observability_outbox(dir)` | durable JSONL outbox + OTLP | none |
| `.pricing(table)` | cost in usage reporting | `PricingTable::DEFAULT` |

See the [reference](../reference/harness-builder.md) for the full list (verifier, planner /
evaluator agents, VCS provider, metric evaluator, tool-call repair, compaction settings, hooks).

## Notes

- **`.build()`** finalizes into a `StandardHarness`. Use `.build_config()` instead if you want the
  `HarnessConfig` without constructing the harness.
- **Order doesn't matter** — each setter is an independent slot. Override the same slot twice and
  the last call wins.
- **Sandbox is the gate.** Catalogue file/exec tools are inert under the default `NullSandbox`;
  give them a `WorkspaceScopedSandbox` to operate on a real directory.
