# Conversation

> Narrative — no code. Rust code: [rust/conversation](../rust/conversation.md). Concept
> background: [memory](../concepts/memory.md).

The harness drives **one task to completion** per `run()` and keeps no state between runs. A
multi-turn conversation is therefore a sequence of runs that share history. There are two ways to
carry that history: thread it yourself, or let the harness persist it.

## Thread the state yourself

The harness is stateless between runs, but `RunResult` hands the conversation back to you:

1. Start with an empty `SessionState` and one session id for the whole conversation.
2. For each user message, run a task over that session, passing the current `SessionState` in.
   The harness appends the new user message to the history before calling the model, so the model
   sees the whole conversation.
3. On success, take the **post-run `SessionState`** off the `RunResult` and keep it for the next
   turn.

That returned state is **lossless**: it includes the assistant tool-call messages and tool-result
messages the loop produced, not just the final text. So this works for tool-using agents, not
just plain chat — feeding it back reconstructs nothing and drops nothing. (An earlier design
rebuilt history from the output string and silently lost tool messages; that's why the post-run
state exists.)

This is the right pattern for a REPL or any single-process chat loop.

## Let the harness persist it

If you'd rather not thread state through your own code, wire a **session store** and turn on
**auto-persist**. Then:

- The harness **loads** the prior state for a session id at the start of a run and **saves** the
  post-run state at the end.
- You reuse the **session id** across runs instead of passing `SessionState` around.

Because the state lives in the store, this survives process restarts — which is exactly what a
web service wants: a request arrives with a session id, the harness loads that conversation,
runs a turn, and persists the result, with no in-memory state to lose. Choose a session store
backend (in-memory for tests, filesystem or a database for real use) from the
[storage seam](../reference/storage-seam.md).

## Which one?

| You're building… | Do this |
|------------------|---------|
| A REPL / single-process chat | **Thread `SessionState`** from each result into the next run |
| A service that resumes across restarts | **Session store + auto-persist**, reuse the id |

Both reuse **one session id** for the conversation — that id is the thread's identity and the key
episodic memory is scoped by.

## See it in action

[`examples/rust/02-conversational-repl`](../../examples/rust/02-conversational-repl) is an
interactive chat loop that threads the returned `SessionState` forward each turn.

## Next steps

- Persist more than the conversation → [memory](../concepts/memory.md),
  [storage-seam](../reference/storage-seam.md).
- Stream tokens to the UI as they arrive → [stream-events](../reference/stream-events.md).
