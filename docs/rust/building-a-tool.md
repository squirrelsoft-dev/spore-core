# Building a tool — Rust

> Rust code for the [building-a-tool guide](../guides/building-a-tool.md). Full runnable program:
> [`examples/rust/03-tool-use`](../../examples/rust/03-tool-use).

## Path 1 — catalogue tools

Pick a ready-made set and hand it to the builder. File tools need a real sandbox:

```rust
use std::sync::Arc;
use spore_core::{HarnessBuilder, OllamaModelInterface, StandardTools};
use spore_core::sandbox::{WorkspaceScopedSandbox, WorkspaceConfig};

let workspace: Arc<WorkspaceScopedSandbox> = /* a sandbox rooted at your workspace */;
let harness = HarnessBuilder::conversational(OllamaModelInterface::new("llama3.2"))
    .sandbox(workspace)
    .tools(StandardTools::coding_set())
    .build();
```

`.tool(t)` adds one tool; `.tools(iter)` adds many. The default `NullSandbox` denies environment
access, so swapping in `WorkspaceScopedSandbox` is what makes the file tools live.

## Path 3 — implement the registry directly

For pure-compute tools, implement `HarnessToolRegistry` (the harness-loop tool registry,
re-exported as `HarnessToolRegistry`). `schemas()` is what the model sees; `dispatch()` runs the
call. No sandbox required.

```rust
use std::sync::Arc;
use spore_core::harness::BoxFut;
use spore_core::{
    HarnessBuilder, HarnessToolRegistry, OllamaModelInterface,
    ToolCall, ToolOutput, ToolSchema,
};

struct LocalTools;

impl HarnessToolRegistry for LocalTools {
    fn schemas(&self) -> Vec<ToolSchema> {
        vec![ToolSchema {
            name: "calculator".to_string(),
            description: "Compute a binary arithmetic op. 'op' is one of + - * /.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "a":  { "type": "number" },
                    "b":  { "type": "number" },
                    "op": { "type": "string", "enum": ["+", "-", "*", "/"] }
                },
                "required": ["a", "b", "op"],
            }),
        }]
    }

    fn dispatch<'a>(&'a self, call: ToolCall) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            match calculator(&call.input) {
                Ok(content) => ToolOutput::Success { content, truncated: false },
                Err(message) => ToolOutput::Error { message, recoverable: true },
            }
        })
    }
}

fn calculator(input: &serde_json::Value) -> Result<String, String> {
    let a = input.get("a").and_then(|v| v.as_f64()).ok_or("missing number 'a'")?;
    let b = input.get("b").and_then(|v| v.as_f64()).ok_or("missing number 'b'")?;
    let op = input.get("op").and_then(|v| v.as_str()).ok_or("missing string 'op'")?;
    let value = match op {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        "/" if b == 0.0 => return Err("division by zero".to_string()),
        "/" => a / b,
        other => return Err(format!("unknown op '{other}'")),
    };
    Ok(value.to_string())
}
```

Wire it in by overriding just the tool registry — every other default stays:

```rust
let harness = HarnessBuilder::conversational(OllamaModelInterface::new("llama3.2"))
    .tool_registry(Arc::new(LocalTools))
    .build();
```

## Notes

- **`ToolOutput`** is `Success { content, truncated }` or `Error { message, recoverable }`. A
  recoverable error is appended to the conversation as a failed tool result, so its message
  steers the next turn — make it informative.
- **Tolerant parsing.** Models sometimes pass numbers as JSON strings (`"144"`). The
  [example](../../examples/rust/03-tool-use) accepts either by falling back to
  `value.as_str().and_then(|s| s.parse().ok())`.
- **Schemas are advertised every turn** and sorted by name before reaching the model — you don't
  manage cache stability yourself.
- **Watching the loop.** Pass a stream sink via `HarnessRunOptions::with_stream(...)` and match
  `HarnessStreamEvent::TurnStart { turn }` to print each Think step. See
  [stream-events](../reference/stream-events.md).

## Run it

```sh
cd examples/rust/03-tool-use
cargo run -- --prompt "what is 144 divided by 12, and what is 'harness' reversed?"
```
