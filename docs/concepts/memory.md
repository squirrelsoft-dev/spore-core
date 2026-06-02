# Memory

> Language-agnostic — no code. For the storage API, see the
> [storage-seam reference](../reference/storage-seam.md); for threading state across turns, the
> [conversation guide](../guides/conversation.md).

Memory in spore-core comes in two layers that are easy to confuse but serve different jobs:
the **conversation state** that flows turn-to-turn, and the **persistent stores** behind the
storage seam.

## SessionState — the conversation

`SessionState` is the message history: the user, assistant, tool-call, and tool-result messages
that make up the conversation so far. It is what the context manager assembles into each turn.

The harness is **stateless between runs**, so `SessionState` is how a conversation continues:

- You pass an optional starting `SessionState` into a `run()`.
- The harness appends the new turn's messages and drives the loop.
- On success (and on failure), `RunResult` hands the **post-run `SessionState`** back to you —
  including the tool-call and tool-result messages the loop produced.

That last point matters: you resume **losslessly** by feeding the returned state into the next
run, with no reconstruction from the final text. (Earlier designs reconstructed history from the
output string, which silently dropped tool messages; that limitation is gone.) See the
[conversation guide](../guides/conversation.md) for the threading pattern and the hands-free
auto-persist alternative.

## The storage seam

Everything durable lives behind a single **storage provider**, which is really four independent
stores, each swappable on its own:

| Store | Holds | Keyed by |
|-------|-------|----------|
| **Session store** | Pause/resume state (`PausedState`) | session id |
| **Memory store** | Episodic & semantic memory entries | scope + session |
| **Run store** | Per-run structured blobs (opaque JSON) | session id + key |
| **Observability store** | Trace spans | session id (queried by time) |

The default provider is all-no-op — nothing persists until you wire a real one. spore-core ships
in-memory and filesystem implementations, and you can compose the four slots from different
backends (e.g. filesystem sessions, a database for memory). See the
[storage-seam reference](../reference/storage-seam.md).

## Episodic vs semantic memory

The memory store distinguishes two kinds of knowledge, tracked by **scope**:

- **Episodic memory** — *what happened in this session*. Scoped to the session id. The running
  log of observations from a particular workspace or thread.
- **Semantic memory** — *what we've learned across sessions*. Distilled rules, patterns, and
  skills. Scoped to the project, falling back to a global scope.

A **merged read** combines the broader scopes (user ∪ project), newest-first, so a task sees
both the local history and the accumulated cross-session knowledge. Episodic memory that recurs
across many sessions is a candidate for distillation into semantic memory — the seam of the
improvement flywheel described in the [spec](../harness-engineering-concepts.md).

Memory is mediated by the harness, not written by the agent directly: writes are validated, and
machine-proposed memories start in a pending-review state rather than going live unreviewed.

## Auto-persist: the hands-free path

If you wire a session store and turn on **auto-persist**, the harness loads and saves session
state by id automatically. You then reuse a session id across runs — even across process restarts
— instead of threading `SessionState` yourself. This is the natural fit for a web service. The
mechanics are in the [conversation guide](../guides/conversation.md) and the
[harness-builder reference](../reference/harness-builder.md).

## Which do I use?

- **Multi-turn chat in one process** → thread `SessionState` from each `RunResult` into the next
  run.
- **A service that resumes across restarts** → wire a session store + auto-persist, reuse the id.
- **Knowledge that should outlive a session** → episodic/semantic memory in the memory store.
