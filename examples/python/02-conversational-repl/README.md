# 02 — conversational REPL

An interactive chat loop where the agent **remembers earlier turns** in the
session. Builds directly on [`01-hello-agent`](../01-hello-agent): same
`HarnessBuilder.conversational(model)` harness — the new idea is *conversation
continuity across `run()` calls*.

## What it shows

- **Multi-turn memory.** The harness is stateless between `run()` calls: each
  call takes an optional starting `SessionState` (the message history) and runs
  one task. The caller owns the conversation.
- **Lossless threading (issue #102).** `RunResultSuccess` now hands the post-run
  `SessionState` back, so after each turn we feed `result.session_state` straight
  into the next run via `HarnessRunOptions(task, session_state=...)`. The harness
  appends the new user line on top of that history before calling the model, so
  the model sees the whole exchange — no reconstruction. This carries tool-call
  and tool-result messages too, which the old "rebuild history from `output`"
  trick could not recover.
- **Hands-free alternative.** Wire a `SessionStore` and call
  `HarnessBuilder.auto_persist_sessions(True)` to auto-load/auto-persist by
  `session_id` instead of threading state by hand.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
```

Plus [`uv`](https://docs.astral.sh/uv/).

## Run

`harness.run` is async; the program drives the REPL with `asyncio.run` and reads
stdin off the event loop so async I/O is never starved.

```sh
cd examples/python/02-conversational-repl
uv run main.py                  # or: uv run main.py --model <id>
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
