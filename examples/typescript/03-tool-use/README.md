# 03 — ReAct with local tools

The first example where the agent **acts**. It thinks, calls a tool, observes the
result, and loops until it can answer — the **Think -> Act -> Observe** cycle.

The tools are deliberately trivial — `calculator`, `get_current_time`,
`reverse_string` — because the star here is the *loop*, not the tools.

## What it shows

- **Implementing the tool registry directly.** `LocalTools` implements the
  harness-loop `ToolRegistry`: `schemas()` advertises the tools to the model and
  `dispatch()` runs them. Because the tools are pure functions (no filesystem),
  the `conversational(model)` defaults — including `NullSandbox` — are fine; we
  only override the registry with `.toolRegistry(...)`.
- **The loop, made visible.** The program prints each turn (`think · turn N`) via
  the stream sink (`on_stream`, keying on `turn_start`) and each tool call +
  result (`act -> name(args) = result`) inside `dispatch`, so you can watch the
  agent work:

  ```
  think  · turn 1
      act    -> calculator({"a":144,"b":12,"op":"/"}) = 12
      act    -> get_current_time({}) = 04:41:03 UTC
      act    -> reverse_string({"text":"harness"}) = ssenrah
  think  · turn 2

  answer (2 turn(s)): 144 divided by 12 is 12. ...
  ```

- **Real-model robustness.** Local models often pass numbers as JSON strings
  (`"144"` not `144`), so `calculator` accepts either. Keeping tools forgiving of
  argument shape is part of making a harness reliable.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
```

## Run

```sh
cd examples/typescript/03-tool-use
pnpm install
pnpm start
pnpm start -- --prompt "reverse the word 'mycelium' and multiply 6 by 7"
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint.

## A note on streamed tool calls

This example attaches a stream sink (to print the `think · turn N` lines), which
drives the turn through the agent's streaming path. That path is fully
tool-aware: the model `StreamEvent` emits a `tool_use_start` event carrying the
call's **id** and **name** at block start, followed by `tool_use_delta` events
for the **arguments** — so a tool call reconstructed from the stream is identical
to one from the non-streaming path, and dispatch routes by the real name. (A
small local model may still sometimes answer from its own reasoning instead of
calling the tool — that's model behavior, not a streaming limitation.)

## What's next

Swap these hand-rolled tools for the built-in catalogue (`.tools(...)`) to give the
agent a workspace-scoped sandbox and real file tools — the loop is identical; only
the registration changes.
