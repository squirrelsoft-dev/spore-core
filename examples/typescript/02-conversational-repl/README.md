# 02 — conversational REPL

An interactive chat loop where the agent **remembers earlier turns** in the
session. Builds directly on [`01-hello-agent`](../01-hello-agent): same
`HarnessBuilder.conversational(model)` harness — the new idea is *conversation
continuity across `run()` calls*.

## What it shows

- **Multi-turn memory.** The harness is stateless between `run()` calls: each
  call takes an optional starting `SessionState` (the message history) and runs
  one task. The caller owns the conversation.
- **Threading session state.** Each successful `RunResult` hands the post-run
  `SessionState` back (issue #102), so after each turn we carry it straight into
  the next run via `HarnessRunOptions.session_state` — lossless, no
  reconstruction. The harness appends the new user line on top before calling the
  model, so the model sees the whole exchange.
- **Why the returned state matters.** A tool-using agent also produces
  tool-call/result messages that the agent's `output` text alone can't recover;
  carrying the returned `SessionState` forward keeps those too. Prefer it
  hands-free? Wire a `StorageProvider` and `HarnessBuilder.autoPersistSessions(true)`
  to auto-load/auto-persist by `session_id` instead of threading state yourself.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
```

## Run

```sh
cd examples/typescript/02-conversational-repl
pnpm install                    # links @spore/core
pnpm start                      # or: pnpm start -- --model <id>
```

Then chat. `/exit` (or `Ctrl-D`) quits. Example:

```
you> My name is Scott.
bot> Hello Scott! It's nice to meet you. ...
you> What is my name?
bot> Your name is Scott. You mentioned it earlier ...
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint.

## What's next

[`03-tool-use`](../03-tool-use/README.md) — the first example where the agent
**acts**: it calls tools in a Think → Act → Observe loop until it can answer.
