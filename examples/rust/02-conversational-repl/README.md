# 02 — conversational REPL

An interactive chat loop where the agent **remembers earlier turns** in the
session. Builds directly on [`01-hello-agent`](../01-hello-agent): same
`HarnessBuilder::conversational(model)` harness — the new idea is *conversation
continuity across `run()` calls*.

## What it shows

- **Multi-turn memory.** The harness is stateless between `run()` calls: each
  call takes an optional starting `SessionState` (the message history) and runs
  one task. The caller owns the conversation.
- **Threading session state.** After each turn we append the user's line and the
  agent's reply to a local `Vec<Message>` and pass it back via
  `HarnessRunOptions::with_session_state(...)`. The harness appends the new user
  line on top before calling the model, so the model sees the whole exchange.
- **An honest limitation.** `RunResult::Success` returns the agent's `output` but
  *not* the post-run `SessionState`. For a no-tools chat agent that's fine — the
  assistant turn is exactly `output`, so reconstruction is exact. A tool-using
  agent would also produce tool-call/result messages that `output` alone can't
  recover; that's motivation for `Success` to return the final state someday.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
```

## Run

```sh
cd examples/rust/02-conversational-repl
cargo run                       # or: cargo run -- --model <id>
```

Then chat. `/exit` (or `Ctrl-D`) quits. Example:

```
you> My name is Scott and my favorite color is teal.
bot> Nice to meet you, Scott! ...
you> What is my name and favorite color?
bot> Your name is Scott, and your favorite color is teal.
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint.

## What's next

[`03-tool-use`](../03-tool-use/README.md) — the first example where the agent
**acts**: it calls tools in a Think → Act → Observe loop until it can answer.
