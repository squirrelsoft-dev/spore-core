# 02 — conversational REPL

An interactive chat loop where the agent **remembers earlier turns** in the
session. Builds directly on [`01-hello-agent`](../01-hello-agent): same
`observability.NewConversationalHarness(model)` harness — the new idea is
*conversation continuity across `Run()` calls*.

## What it shows

- **Multi-turn memory.** The harness is stateless between `Run()` calls: each
  call takes an optional starting `SessionState` (the message history) and runs
  one task. The caller owns the conversation.
- **Lossless threading (issue #102).** On success, `RunResult.SessionState`
  carries the full post-run history — this turn's user message, the assistant
  reply, and (for tool-using agents) any tool-call/result messages. We feed it
  straight into the next run via `HarnessRunOptions.SessionState`; the harness
  appends the new user line on top before calling the model, so the model sees
  the whole exchange. No reconstruction from `Output`.
- **Hands-free alternative.** Wire a `SessionStore` and set
  `AutoPersistSessions` on the harness config to auto-load/auto-persist by
  `SessionID` — then you reuse the id instead of threading state at all.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
```

## Run

```sh
cd examples/go/02-conversational-repl
go run .                        # or: go run . --model <id>
```

Then chat. `/exit` (or `Ctrl-D`) quits. Example:

```
you> My name is Scott and my favorite color is teal.
bot> Nice to meet you, Scott! ...
you> What is my name and favorite color?
bot> Your name is Scott, and your favorite color is teal.
```

Or pipe a scripted conversation:

```sh
printf "My name is Scott.\nWhat is my name?\n/exit\n" | go run .
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint.

## What's next

[`03-tool-use`](../03-tool-use/README.md) — the first example where the agent
**acts**: it calls tools in a Think → Act → Observe loop until it can answer.
