# Quickstart

> Narrative — no code. The runnable code for your language is on the matching page:
> [rust](../rust/quickstart.md) · [ts](../typescript/quickstart.md) · [py](../python/quickstart.md) · [go](../go/quickstart.md).

The goal of the quickstart is the smallest real thing: turn a model into a running agent and get
a response back. No tools, no filesystem, no multi-turn state — just `Model → Harness → Task →
Result`.

## The five-line shape

1. **Construct a model interface.** Point it at a provider — a local Ollama model is the
   zero-setup choice for trying things out; swap in a hosted provider later without touching the
   rest of the code.
2. **Build a conversational harness from the model.** The `conversational(model)` preset fills in
   every required component — a model-backed agent, an empty tool registry, a null sandbox, a
   standard context manager, and respond-and-stop termination — so you don't assemble them by
   hand.
3. **Create a task** from a single instruction string.
4. **Run it** and await the `RunResult`.
5. **Match the result.** On success you get the output text, the turn count, token usage, and the
   post-run session state.

That's the whole path. Everything else in spore-core is overriding one of the defaults the
preset chose for you.

## Running a local model

The examples default to Ollama because it runs locally with no API key:

```sh
ollama serve &          # start the local model server
ollama pull llama3.2    # pull a small default model
```

Then run your language's quickstart program. The model id and endpoint are read from arguments
or environment variables so you can swap models without recompiling.

## What you get back

`RunResult` is an enum. The happy path is **Success**, carrying:

- `output` — the model's final text,
- `turns` — how many model calls it took (one, for the quickstart),
- `usage` — aggregated token counts,
- `session_state` — the post-run conversation, ready to feed into a follow-up run.

The other variants — **Failure** (with a typed reason), **WaitingForHuman** (a paused state to
resume), and **Escalate** (a tool signalled the caller to take over) — matter as you add tools,
approvals, and longer loops. For the quickstart, match `Success` and print the output.

## See it in action

Run [`examples/rust/01-hello-agent`](../../examples/rust/01-hello-agent) — the smallest live
harness, end to end.

## Next steps

- Give the agent something to do → [building-a-tool](./building-a-tool.md).
- Make it remember across turns → [conversation](./conversation.md).
- Swap the preset's defaults for your own components → [custom-harness](./custom-harness.md).
