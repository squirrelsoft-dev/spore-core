# 01 — hello agent

The smallest real thing you can build with spore-core.

This example stands up a harness from its five required components, points it at a
live local model, and runs a single turn whose only job is to say hello — no tools,
no observability, no multi-turn state. It is the on-ramp to everything else in this
directory.

## What it shows

- The few-lines path: `observability.NewConversationalHarness(model)` defaults every
  required component (a model-backed agent, an empty tool registry, a null sandbox,
  a standard context manager, and respond-and-stop termination), so you go from a
  model to a running harness in one call. (It lives in the `observability` package
  rather than the root `sporecore` package because assembling the standard context
  manager would otherwise form a `sporecore` → `contextmgr` import cycle.)
- `sporecore.SimpleTask(instruction)` — a one-shot task with a fresh session and a
  default loop strategy (ReAct, 8 max iterations).
- Where the later examples plug in: every default above is overridable through the
  conversational builder. [`02-conversational-repl`](../02-conversational-repl/README.md)
  threads multi-turn session state; [`03-tool-use`](../03-tool-use/README.md) adds tools.

## Prerequisites

A local [Ollama](https://ollama.com) server with a pulled model:

```sh
ollama serve &          # start the model server
ollama pull llama3.2    # the default model this example asks for
```

## Run

```sh
cd examples/go/01-hello-agent
go run .                       # uses llama3.2 at http://localhost:11434
go run . --model qwen2.5       # or any other model you've pulled
```

The example modules are part of `go/go.work`, so `go run .` resolves the local
`spore-core` module directly — no published version or `replace` directive needed.

Environment overrides (both optional):

- `SPORE_OLLAMA_MODEL` — default model id (overridden by `--model`).
- `SPORE_OLLAMA_BASE_URL` — Ollama endpoint (default `http://localhost:11434`).

Expected output is a one-line greeting and `Success (1 turn(s))` (occasionally 2).

## What's next

[`02-conversational-repl`](../02-conversational-repl/README.md) — an interactive chat
loop where the agent remembers earlier turns by threading session state across
`Run()` calls.
