# Reference: the storage seam

> `StorageProvider` and the four domain stores. Language-agnostic. Concept background in
> [memory](../concepts/memory.md); wiring it up in [conversation](../guides/conversation.md) and
> [rust/conversation](../rust/conversation.md).

Everything durable in a harness lives behind one **storage provider**, which is really four
independent stores. Each is a separate trait, so you can back them with different technologies —
filesystem sessions, a database for memory, an OTLP pipe for spans — or all from one backend.

## The four stores

| Store | Purpose | Keyed by | Shape |
|-------|---------|----------|-------|
| **Session store** | Pause/resume lifecycle state | session id | get / put / delete / list |
| **Memory store** | Episodic & semantic memory entries | scope + session | append / read-most-recent / merged-read |
| **Run store** | Per-run structured blobs (opaque JSON the store doesn't interpret) | session id + key | get / put / delete / list-keys |
| **Observability store** | Trace spans | session id (queried by time range) | append / get-spans / query-sessions / flush |

The session store is what `auto_persist_sessions` reads and writes. The memory store holds the
knowledge that outlives a session. The run store is a general key/value scratchpad scoped to a
run. The observability store is append-only and queried by session and time, not by key.

## Memory scopes

Memory entries carry a **scope**: `User`, `Project`, or `Local`.

- A **read** returns the most-recent N entries for a given scope, newest-first.
- A **merged read** returns `User ∪ Project`, newest-first by timestamp, with no de-duplication.
  `Local` is excluded from the merge in v1.

The merge algorithm has exactly one implementation, inherited by every backend (real, in-memory,
no-op, test mock) — so merged reads behave identically no matter what's underneath. This is the
mechanism behind episodic-vs-semantic memory described in [memory](../concepts/memory.md): a task
sees both its own session history and the accumulated cross-session knowledge in one read.

## Composing a provider

| Constructor | Use |
|-------------|-----|
| **`single(backend)`** | Clone one backend that implements all four traits into all four slots. |
| **`new(session, memory, run, observability)`** | Supply four explicit per-domain stores. |
| **`no_op()`** | All-no-op. The default when storage is never set — nothing persists. |
| **composite builder** | Mix and match: route each domain (and each memory scope) to its own backend. |

## Built-in backends

| Backend | Persistence | Use for |
|---------|-------------|---------|
| **No-op provider** | none | Tests, or runs that shouldn't persist anything (the default). |
| **In-memory provider** | process lifetime | Tests and short-lived processes; lost on exit. |
| **Filesystem provider** | on disk under a root | Local development, single-node deployments; survives restarts. |

For production, compose the slots: e.g. a filesystem or database session store for resumable
conversations, a database memory store for shared knowledge, and an OTLP-forwarding observability
provider (see [observability](../concepts/observability.md)).

## Errors

Store operations return a typed storage error rather than panicking, so a backend failure
(missing file, unreachable database) is a value the caller handles, not a crash. Reads that find
nothing return an empty/absent value, not an error.

## How it connects to the harness

- `auto_persist_sessions(true)` makes the harness use the **session store** to load prior state
  at run start and save post-run state at the end, keyed by session id — the hands-free
  conversation path.
- Memory tools and the memory subsystem read and write the **memory store** (mediated by the
  harness; the agent doesn't write memory directly).
- The observability provider writes spans through the **observability store** as the loop runs.

The provider defaults to all-no-op, so a harness persists nothing until you attach a real one via
the builder's `storage(...)` method — see the
[harness-builder reference](./harness-builder.md#storage--sessions).
