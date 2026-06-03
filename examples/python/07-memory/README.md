# 07 — Memory (the storage seam, as a markdown file)

The harness is **stateless**. Every byte of durable state — sessions, run
state, traces, and memory — lives behind one injectable seam: the
`StorageProvider`. Memory is just one of its four domains (`MemoryStore`), and
**you implement it**. The simplest useful implementation is a human-readable
markdown file.

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

- A `MarkdownMemoryProvider` — a concrete `MemoryStore` that reads/writes a
  `.md` file (`memory_provider.py`).
- **Run 1 (`--phase store`):** the agent is given the Project Ironwood briefing
  and writes each fact to `memory.md` via the built-in `memory` tool.
- The process exits. `memory.md` is a visible, inspectable artifact — open it.
- **Run 2 (`--phase recall`):** a fresh process loads `memory.md` through the
  same provider and answers questions that **restate none of the facts**. The
  agent recalls them from memory.
- The `StorageProvider` / `MemoryStore` protocol as the clean swap point.

## `StorageProvider` is the swap point

`main` never touches `append_memory` / `get_memories`. It composes the provider
and hands it to the builder:

```python
storage = MarkdownMemoryProvider(memory_path).into_storage_provider()

harness = (
    HarnessBuilder.conversational(model)
    .storage(storage)               # ← the seam
    .tool(StandardTools.memory())   # ← the built-in memory read/write tool
    .system_prompt(system_prompt)
    .build()
)
```

The harness threads `storage.memory()` into the `memory` tool's context on
every run; the **agent** drives all reads and writes from inside the ReAct
loop. Swap the provider and nothing else changes:

| Provider                          | On-disk shape     | Same interface?     |
| --------------------------------- | ----------------- | ------------------- |
| `FileSystemStorageProvider`       | JSONL append log  | yes (`MemoryStore`) |
| `MarkdownMemoryProvider` (here)   | readable markdown | yes (`MemoryStore`) |
| your database / KV / S3 impl      | whatever you want | yes (`MemoryStore`) |

`into_storage_provider()` composes the real markdown `MemoryStore` for the
memory domain and `NoOpStorageProvider` for the other three (session, run,
observability) — this example only needs memory.

## The pinned session id (don't skip this)

Memory is keyed by `SessionId`, and the `memory` tool always uses the run's
session id. So **both phases must use the same id** or recall reads an empty
session. This example pins:

```python
SESSION = SessionId("project-ironwood")   # NOT new_session_id()
```

`new_session_id()` would give Run 2 a different key and it would recall nothing.
The provider also stamps the session id into each `memory.md` block, so one file
can hold several sessions without cross-talk.

All facts use `StorageScope.PROJECT` (the `memory` tool rejects `Local`), and
the prompts tell the agent to read/write with `scope: "project"` consistently.

## What `memory.md` looks like

Each `MemoryEntry` becomes one readable block. The header carries the
round-trip fields (scope, session, timestamp, role); the body is the content:

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

`append_memory` writes these blocks; `get_memories` parses them back, filters
by scope + session, sorts newest-first by timestamp, and takes `limit`. A
missing file reads as empty, and a **hand-edited** file is tolerated — prose
before the first block and human-added `##` headings are ignored rather than
swallowed into an entry. (See the unit tests in `test_memory_provider.py`.)

## Run it

```sh
ollama serve &
ollama pull llama3.2

cd examples/python/07-memory
uv run main.py --phase store     # agent writes facts → memory.md
cat memory.md                    # inspect the human-readable artifact
uv run main.py --phase recall    # agent answers from memory.md alone
```

Running with **no flag** runs `store` and then points you at `recall`:

```sh
uv run main.py                   # = --phase store, then prints the next step
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and
endpoint, and `--model <id>` overrides the id for a single run — same as the
earlier examples. See `.env.example`.

`memory.md` is written next to the sources and is `.gitignore`d. Delete it by
hand to reset between demo runs.

## Tests

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## A note on the model

This uses Ollama (`llama3.2`) like examples 01–06, so it runs with no hosted
account. Recall quality scales with the model: a **larger hosted model follows
the store/recall tool protocol more reliably**. The harness is model-agnostic —
swap the model interface and change nothing else.

### Tool-calling mode: native by default, `--structured` to opt in

By default this example uses **native Ollama tool calling** — the real typed
tool schema — which works well for tool-capable / cloud models like
`gemma4:31b-cloud`. Pass `--structured` to opt into schema-constrained decoding
via `HarnessBuilder.model_params(ModelParams(structured_tool_calls=True))`, which
helps **small local models** (e.g. `llama3.2`) emit one clean `memory` tool call
per turn instead of malformed JSON:

```sh
uv run main.py --phase store                # native tool calling (default)
uv run main.py --phase store --structured   # constrained decoding for small models
```

One caveat for structured mode: it exposes an always-available `final` envelope
whose content is optional, so a **capable** model can emit `{"tool":"final"}`
prematurely and hand back an EMPTY answer without ever calling the `memory` tool.
If you run with `--structured` and see an empty answer and no `memory.md` on disk,
drop the flag and use the native default.
