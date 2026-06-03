# 07 — Memory (the storage seam, as a markdown file)

The harness is **stateless**. Every byte of durable state — sessions, run
state, traces, and memory — lives behind one injectable seam: the
`StorageProvider`. Memory is just one of its four domains (`MemoryStore`), and
**you implement it**. The simplest useful implementation is a human-readable
markdown file.

This example ships a `MarkdownMemoryProvider` and runs the _same_ agent twice
against it: once to **store** facts, once — in a fresh process — to **recall**
them with nothing in the prompt.

## The CLAUDE.md analogy

If you use Claude Code, you already know this pattern. `CLAUDE.md` is a plain
markdown file the agent reads at the start of a session — durable, human-
readable working memory that survives process restarts and that you can open
and edit by hand. **`memory.md` is to this agent what `CLAUDE.md` is to Claude
Code.** Same idea, made explicit: structured markdown as the agent's long-term
memory. The difference here is that the agent _writes_ it too, through a tool,
one fact at a time.

## What it demonstrates

- A `MarkdownMemoryProvider` — a concrete `MemoryStore` that reads/writes a
  `.md` file (`src/memory-provider.ts`).
- **Run 1 (`--phase store`):** the agent is given the Project Ironwood briefing
  and writes each fact to `memory.md` via the built-in `memory` tool.
- The process exits. `memory.md` is a visible, inspectable artifact — open it.
- **Run 2 (`--phase recall`):** a fresh process loads `memory.md` through the
  same provider and answers questions that **restate none of the facts**. The
  agent recalls them from memory.
- The `StorageProvider`/`MemoryStore` interface as the clean swap point.

## `StorageProvider` is the swap point

`main` never touches `appendMemory`/`getMemories`. It composes the provider and
hands it to the builder:

```ts
const storage = new MarkdownMemoryProvider(memoryPath).intoStorageProvider();

const harness = HarnessBuilder.conversational(model)
  .storage(storage) // ← the seam
  .tool(StandardTools.memory()) // ← the built-in memory read/write tool
  .systemPrompt(systemPrompt)
  .build();
```

The harness threads `storage.memory()` into the `memory` tool's context on
every run; the **agent** drives all reads and writes from inside the ReAct
loop. Swap the provider and nothing else changes:

| Provider                        | On-disk shape    | Same interface?     |
| ------------------------------- | ---------------- | ------------------- |
| `FileSystemStorageProvider`     | JSONL append log | yes (`MemoryStore`) |
| `MarkdownMemoryProvider` (here) | readable markdown | yes (`MemoryStore`) |
| your database / KV / S3 impl    | whatever you want | yes (`MemoryStore`) |

`intoStorageProvider()` composes the real markdown `MemoryStore` for the memory
domain and `NoOpStorageProvider` for the other three (session, run,
observability) — this example only needs memory.

## The pinned session id (don't skip this)

Memory is keyed by `SessionId`, and the `memory` tool always uses
`ctx.sessionId`. So **both phases must use the same id** or recall reads an
empty session. This example pins:

```ts
const SESSION = SessionId.of("project-ironwood"); // NOT SessionId.generate()
```

`SessionId.generate()` would give Run 2 a different key and it would recall
nothing. The provider also stamps the session id into each `memory.md` block, so
one file can hold several sessions without cross-talk.

All facts use scope `"project"` (the `memory` tool rejects `"local"`), and the
prompts tell the agent to read/write with `scope: "project"` consistently.

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

`appendMemory` writes these blocks; `getMemories` parses them back, filters by
scope + session, sorts newest-first by timestamp, and takes `limit`. A missing
file reads as empty, and a **hand-edited** file is tolerated — prose before the
first block and human-added `##` headings are ignored rather than swallowed into
an entry. (See the unit tests in `tests/memory-provider.test.ts`.)

## Run it

```sh
ollama serve &
ollama pull llama3.2

pnpm install
pnpm start -- --phase store     # agent writes facts → memory.md
cat memory.md                   # inspect the human-readable artifact
pnpm start -- --phase recall    # agent answers from memory.md alone
```

Running with **no flag** runs `store` and then points you at `recall`:

```sh
pnpm start                      # = --phase store, then prints the next step
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

### Native vs. structured tool calls (`--structured`)

By default the example uses **native** Ollama tool calling: the `memory` tool
gets a real typed schema and there is no always-on `final` envelope, so
tool-capable models (including hosted `*-cloud` models such as
`gemma4:31b-cloud`) follow the store/recall protocol reliably.

Small local models (e.g. `llama3.2`) sometimes leak `<|python_tag|>` or emit
malformed JSON on the native channel. Pass `--structured` to switch on
schema-constrained decoding
(`modelParams({ structured_tool_calls: true, stop_sequences: [] })`), which
forces one clean `memory` tool call per turn:

```sh
pnpm start -- --phase store --structured     # llama3.2 with constrained decoding
pnpm start -- --phase store --model gemma4:31b-cloud   # capable model, native
```

> Note: structured mode exposes an always-available `final` envelope that weaker
> models can take prematurely (ending the turn without storing anything). If a
> phase produces no memory write — an empty answer and no new `memory.md` block —
> drop `--structured`.

## Tests

```sh
pnpm test     # vitest — 9 unit tests over the MarkdownMemoryProvider
pnpm lint
pnpm format   # prettier --check
```
