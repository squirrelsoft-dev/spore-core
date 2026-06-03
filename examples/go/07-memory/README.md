# 07 — Memory (the storage seam, as a markdown file)

The harness is **stateless**. Every byte of durable state — sessions, run
state, traces, and memory — lives behind one injectable seam: the
`storage.StorageProvider`. Memory is just one of its four domains
(`storage.MemoryStore`), and **you implement it**. The simplest useful
implementation is a human-readable markdown file.

This example ships a `MarkdownMemoryProvider` and runs the *same* agent twice
against it: once to **store** facts, once — in a fresh process — to **recall**
them with nothing in the prompt.

## The CLAUDE.md analogy

If you use Claude Code, you already know this pattern. `CLAUDE.md` is a plain
markdown file the agent reads at the start of a session — durable, human-
readable working memory that survives process restarts and that you can open
and edit by hand. **`memory.md` is to this agent what `CLAUDE.md` is to Claude
Code.** Same idea, made explicit: structured markdown as the agent's long-term
memory. The difference here is that the agent *writes* it too, through a tool,
one fact at a time.

## What it demonstrates

- A `MarkdownMemoryProvider` — a concrete `storage.MemoryStore` that
  reads/writes a `.md` file (`memory_provider.go`).
- **Run 1 (`--phase store`):** the agent is given the Project Ironwood briefing
  and writes each fact to `memory.md` via the built-in `memory` tool.
- The process exits. `memory.md` is a visible, inspectable artifact — open it.
- **Run 2 (`--phase recall`):** a fresh process loads `memory.md` through the
  same provider and answers questions that **restate none of the facts**. The
  agent recalls them from memory.
- The `storage.MemoryStore` interface as the clean swap point.

## `StorageProvider` is the swap point

`main` never touches `AppendMemory`/`GetMemories`. It composes the provider and
hands its memory-domain store to the builder:

```go
storage := NewMarkdownMemoryProvider(memoryPath).IntoStorageProvider()

harness := observability.ConversationalBuilder(mi).
    Storage(nil, storage.Memory()).        // ← the seam: the memory-domain store
    Tool(tools.StandardTools{}.Memory()).  // ← the built-in memory read/write tool
    SystemPrompt(systemPrompt).
    Build()
```

The harness threads the memory store into the `memory` tool's context on every
run; the **agent** drives all reads and writes from inside the ReAct loop. Swap
the provider and nothing else changes:

| Provider                          | On-disk shape      | Same interface?            |
| --------------------------------- | ------------------ | -------------------------- |
| `FileSystemStorageProvider`       | JSONL append log   | yes (`storage.MemoryStore`) |
| `MarkdownMemoryProvider` (here)   | readable markdown  | yes (`storage.MemoryStore`) |
| your database / KV / S3 impl      | whatever you want  | yes (`storage.MemoryStore`) |

`IntoStorageProvider()` composes the real markdown `MemoryStore` for the memory
domain and `storage.NoOpStorageProvider` for the other three (session, run,
observability) — this example only needs memory. (In Go, the builder's
`Storage(runStore, memStore)` seam takes the run-domain and memory-domain stores
directly; we pass `nil` for the run store and the markdown store for memory.)

## The pinned session id (don't skip this)

Memory is keyed by `SessionID`, and the `memory` tool always uses the run's
`SessionID`. So **both phases must use the same id** or recall reads an empty
session. This example pins:

```go
func session() sporecore.SessionID {
    return sporecore.SessionID("project-ironwood") // NOT a generated id
}
```

A generated id (`sporecore.NewSessionID()`) would give Run 2 a different key and
it would recall nothing. The provider also stamps the session id into each
`memory.md` block, so one file can hold several sessions without cross-talk.

All facts use `storage.StorageScopeProject` (the `memory` tool rejects `Local`),
and the prompts tell the agent to read/write with `scope: "project"`
consistently.

## What `memory.md` looks like

Each `MemoryEntry` becomes one readable block. The header carries the round-trip
fields (scope, session, timestamp, role); the body is the content:

```markdown
# Agent Memory

Human-readable working memory for this agent. Each `##` block below is one
remembered entry.

## [project] [project-ironwood] 2026-06-02T12:00:00Z — assistant

Postgres 15 is the system of record; chosen over DynamoDB because relational
invariants on inventory counts mattered more than infinite scale.

## [project] [project-ironwood] 2026-06-02T12:00:01Z — assistant

Hard constraints: p99 latency budget of 50ms for position lookups; no PII may
leave eu-west-1 (GDPR data-residency).
```

`AppendMemory` writes these blocks; `GetMemories` parses them back, filters by
scope + session, sorts newest-first by timestamp, and takes `limit`. A missing
file reads as empty, and a **hand-edited** file is tolerated — prose before the
first block and human-added `##` headings are ignored rather than swallowed into
an entry. (See the unit tests in `memory_provider_test.go`.)

## Run it

```sh
ollama serve &
ollama pull llama3.2

cd examples/go/07-memory
go run . --phase store     # agent writes facts → memory.md
cat memory.md              # inspect the human-readable artifact
go run . --phase recall    # agent answers from memory.md alone
```

Running with **no flag** runs `store` and then points you at `recall`:

```sh
go run .                   # = --phase store, then prints the next step
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and
endpoint, and `--model <id>` overrides the id for a single run — same as the
earlier examples. See `.env.example`.

`memory.md` is written next to the sources and is `.gitignore`d. Delete it by
hand to reset between demo runs.

## A note on the model

This uses Ollama (`llama3.2`) like examples 01–06, so it runs with no hosted
account. Recall quality scales with the model: a **larger hosted model follows
the store/recall tool protocol more reliably**. The harness is model-agnostic —
swap the model interface and change nothing else.
