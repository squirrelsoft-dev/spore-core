# Conversation — Rust

> Rust code for the [conversation guide](../guides/conversation.md). Full runnable program:
> [`examples/rust/02-conversational-repl`](../../examples/rust/02-conversational-repl).

## Thread the state yourself

Build the harness once, reuse it for every turn, and carry the post-run `SessionState` forward:

```rust
use spore_core::{
    Harness, HarnessBuilder, HarnessRunOptions, LoopStrategy, OllamaModelInterface,
    RunResult, SessionId, SessionState, Task,
};

let harness = HarnessBuilder::conversational(OllamaModelInterface::new("llama3.2")).build();

// One session id for the whole conversation; one running state we thread back in.
let session_id = SessionId::generate();
let mut state = SessionState::default();

loop {
    let line = read_user_line();            // your input source; break on EOF / "/exit"
    if line.is_empty() { continue; }

    let task = Task::new(
        line,
        session_id.clone(),
        LoopStrategy::ReAct { max_iterations: 4 },
    );
    // Thread the running state in. The harness appends `line` as the new user
    // message before calling the model.
    let options = HarnessRunOptions::new(task).with_session_state(state.clone());

    match harness.run(options).await {
        RunResult::Success { output, session_state, .. } => {
            println!("bot> {output}");
            // Carry the post-run state forward — lossless, no reconstruction.
            // It already holds this turn's user + assistant (+ tool) messages.
            state = session_state;
        }
        other => eprintln!("bot> [did not succeed: {other:?}]"),
    }
}
```

The key line is `state = session_state;` — destructure the post-run `SessionState` off
`RunResult::Success` and reuse it. `state.messages.len()` tells you how much history has
accumulated.

## Let the harness persist it

Wire a `SessionStore` and enable auto-persist; then reuse the **id** instead of threading state:

```rust
use std::sync::Arc;
use spore_core::{HarnessBuilder, OllamaModelInterface, InMemoryStorageProvider, StorageProvider};

let storage = StorageProvider::single(Arc::new(InMemoryStorageProvider::new()));
let harness = HarnessBuilder::conversational(OllamaModelInterface::new("llama3.2"))
    .storage(Arc::new(storage))
    .auto_persist_sessions(true)
    .build();

// Each turn: same `session_id`, no `with_session_state`. The harness loads the
// prior state and saves the new one automatically.
let task = Task::new(line, session_id.clone(), LoopStrategy::ReAct { max_iterations: 4 });
let _ = harness.run(HarnessRunOptions::new(task)).await;
```

Swap `InMemoryStorageProvider` for `FileSystemStorageProvider` (or a composed provider) to
survive restarts — see [storage-seam](../reference/storage-seam.md).

## Notes

- **`with_session_state(state)`** seeds a run with prior history. Omit it (and rely on
  auto-persist) for the hands-free path.
- **`SessionState::default()`** is an empty conversation. `RunResult::Success.session_state` is
  the post-run conversation, carrying tool-call and tool-result messages too — which is why it
  round-trips a tool-using agent losslessly.
- **Reuse one `SessionId`** for the whole conversation; that id is the thread identity and the
  auto-persist key.

## Run it

```sh
cd examples/rust/02-conversational-repl
cargo run        # then chat; /exit or Ctrl-D to quit
```
