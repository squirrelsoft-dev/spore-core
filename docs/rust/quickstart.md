# Quickstart — Rust

> Rust code for the [quickstart guide](../guides/quickstart.md). Full runnable program:
> [`examples/rust/01-hello-agent`](../../examples/rust/01-hello-agent).

## Dependencies

```toml
[dependencies]
spore-core = { path = "../../rust/crates/spore-core" } # or a published version
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

## The whole thing

A model, a harness, a task — that's the setup. `conversational(model)` defaults every required
component, so this is three lines of wiring:

```rust
use spore_core::{
    Harness, HarnessBuilder, HarnessRunOptions, OllamaModelInterface, RunResult, Task,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A model, a harness, a task.
    let model = OllamaModelInterface::new("llama3.2");
    let harness = HarnessBuilder::conversational(model).build();
    let task = Task::simple("Reply with a friendly one-line greeting.");

    match harness.run(HarnessRunOptions::new(task)).await {
        RunResult::Success { output, turns, .. } => {
            println!("Success ({turns} turn(s)): {output}");
            Ok(())
        }
        other => {
            eprintln!("did not succeed: {other:?}");
            std::process::exit(1);
        }
    }
}
```

## Notes

- **`OllamaModelInterface::new("llama3.2")`** talks to a local Ollama server at
  `http://localhost:11434`. Use `with_base_url(model_id, base_url)` to point elsewhere. Swap in
  `AnthropicModelInterface` / `OpenAIModelInterface` to use a hosted provider — the rest of the
  code is unchanged.
- **`HarnessBuilder::conversational(model)`** returns a builder with every required component
  defaulted (a `ModelAgent`, an `EmptyToolRegistry`, a `NullSandbox`, a `StandardContextManager`,
  and `CompleteOnFinalResponse` termination). `.build()` finalizes it into a `StandardHarness`.
- **`Task::simple(instruction)`** generates a fresh session id and uses a default ReAct loop.
  For an explicit session or loop strategy, use `Task::new(instruction, session_id, strategy)`.
- **`harness.run(...)`** requires the `Harness` trait in scope (the `use` line above). It returns
  a `RunResult`; match `Success` for the happy path. The `..` skips `usage` and `session_state` —
  the [conversation page](./conversation.md) uses the latter to continue the chat.

## Configurable model id

The example reads the model id and endpoint from args/env so you can swap models without a
recompile:

```rust
let model_id = std::env::var("SPORE_OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".to_string());
let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
    .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());
let model = OllamaModelInterface::with_base_url(&model_id, base_url);
```

## Run it

```sh
ollama serve &
ollama pull llama3.2
cd examples/rust/01-hello-agent
cargo run
```
