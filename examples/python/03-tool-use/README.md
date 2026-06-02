# 03 — ReAct with local tools

The first example where the agent **acts**. It thinks, calls a tool, observes the
result, and loops until it can answer — the **Think → Act → Observe** cycle.

The tools are deliberately trivial — `calculator`, `get_current_time`,
`reverse_string` — because the star here is the *loop*, not the tools.

## What it shows

- **Implementing the tool registry directly.** `LocalTools` satisfies the
  harness-loop `ToolRegistry` protocol structurally: `schemas()` advertises the
  tools to the model, `dispatch()` runs them, and `is_always_halt()` reports
  whether a tool halts the loop. Because the tools are pure functions (no
  filesystem), the `conversational(model)` defaults — including the null
  sandbox — are fine; we only override the registry with `.tool_registry(...)`.
- **The loop, made visible.** The program prints each turn (`think · turn N`)
  via the stream sink (`HarnessRunOptions(task, on_stream=...)`) and each tool
  call + result (`act → name(args) = result`), so you can watch the agent work:

  ```
  think  · turn 1
      act    → calculator({"a": "144", "b": "12", "op": "/"}) = 12
      act    → get_current_time({}) = 04:41:03 UTC
      act    → reverse_string({"text": "harness"}) = ssenrah
  think  · turn 2

  answer (2 turn(s)): 144 divided by 12 is 12. The current time is 04:41:03 UTC.
  'harness' reversed is 'ssenrah'.
  ```

- **Real-model robustness.** Local models often pass numbers as JSON strings
  (`"144"` not `144`), so `calculator` accepts either. Keeping tools forgiving of
  argument shape is part of making a harness reliable.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
```

Plus [`uv`](https://docs.astral.sh/uv/).

## Run

```sh
cd examples/python/03-tool-use
uv run main.py
uv run main.py --prompt "reverse the word 'mycelium' and multiply 6 by 7"
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint.

## What's next

A filesystem agent swaps these hand-rolled tools for the built-in catalogue
(`.tools(StandardTools.coding_set())`) so the agent reads, writes, and lists real
files through a workspace-scoped sandbox. The loop is identical; only the
registration and sandbox change.
