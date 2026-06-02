# 01 — hello agent

The smallest real thing you can build with spore-core.

This example stands up a harness from its five required components, points it at a
live local model, and runs a single turn whose only job is to say hello — no tools,
no observability, no multi-turn state. It is the on-ramp: read this before the full
[`e2e_agent.rs`](../../rust/crates/spore-core/examples/e2e_agent.rs).

## What it shows

- The few-lines path: `HarnessBuilder::conversational(model).build()` defaults every
  required component (a model-backed agent, an empty tool registry, a null sandbox,
  a standard context manager, and respond-and-stop termination), so you go from a
  model to a running harness in one call.
- `Task::simple(instruction)` — a one-shot task with a fresh session and a default
  loop strategy.
- Where the later examples plug in: every default above is overridable through the
  builder. `02-react-tools` adds real tools and a workspace sandbox; later examples
  swap the loop strategy, memory, and so on.

## Prerequisites

A local [Ollama](https://ollama.com) server with a pulled model:

```sh
ollama serve &          # start the model server
ollama pull llama3.2    # the default model this example asks for
```

## Run

```sh
cd examples/01-hello-agent
cargo run                 # uses llama3.2 at http://localhost:11434
cargo run -- --model qwen2.5   # or any other tool-capable model you've pulled
```

Environment overrides (both optional):

- `SPORE_OLLAMA_MODEL` — default model id (overridden by `--model`).
- `SPORE_OLLAMA_BASE_URL` — Ollama endpoint (default `http://localhost:11434`).

Expected output is a one-line greeting and `Success (1 turn(s))` (occasionally 2).

## What's next

[`rust/crates/spore-core/examples/e2e_agent.rs`](../../rust/crates/spore-core/examples/e2e_agent.rs)
is the full version: real file/shell tools, multi-turn sessions, live context
compaction, tool-failure recovery, and OTLP observability — selectable via scenario
flags.
