# 04 — Filesystem agent (catalogue tools)

The first example that uses spore-core's **built-in catalogue tools**. The agent
reads a directory of `.txt` files, summarizes each, and writes a `SUMMARY.md`
collecting the summaries.

This is [`03-tool-use`](../03-tool-use/README.md) with one change. The thesis is
that catalogue tools work out of the box and **the harness doesn't change** —
only the registration path does.

## The contrast with 03

|         | 03 — tool-use                                          | 04 — filesystem-agent                              |
| ------- | ------------------------------------------------------ | -------------------------------------------------- |
| Builder | `observability.ConversationalBuilder(model)`           | `observability.ConversationalBuilder(model)` *(same)* |
| Loop    | `StrategyReAct`                                        | `StrategyReAct` *(same)*                           |
| Output  | stream-printed `think` / `act`                         | stream-printed `think` / `act` / `obs` *(same)*    |
| Tools   | hand-built `*sporecore.StandardToolRegistry`           | `.Tools(tools.StandardTools{}.CodingSet()...)`     |
| Sandbox | default `NullSandbox` (pure tools)                     | `WorkspaceScopedSandbox` over `sample-files/`      |

In 03 we built the `*sporecore.StandardToolRegistry` ourselves and registered
each tool by hand. Here the registration collapses to a single builder line:

```go
harness := observability.ConversationalBuilder(mi).
    Sandbox(sandbox).                            // ← the one extra requirement vs 03
    Tools(tools.StandardTools{}.CodingSet()...). // ← read_file, write_file, list_dir, …
    SystemPrompt(systemPrompt).
    Build()
```

### One extra requirement: a real sandbox

03's tools were pure functions, so the conversational default `NullSandbox` was
fine. Catalogue **file** tools go through a sandbox, so 04 wires a
`WorkspaceScopedSandbox` scoped to `sample-files/`. That is the only addition
over 03 — and it is what keeps `read_file` / `write_file` from escaping the
directory.

### Watching the loop

Because the catalogue dispatches internally, the Act / Observe lines come from
harness **stream events** (`HarnessStreamToolCall` carries the name + args,
`HarnessStreamToolResult` carries the result content) rather than from inside a
hand-rolled dispatch:

```
think  · turn 1
    act    → list_dir({"path":"."})
    obs → harness.txt mycelium.txt react.txt spores.txt
    act    → read_file({"path":"mycelium.txt"})
    obs → Mycelium is the root-like network of fungal threads …
    …
    act    → write_file({"path":"SUMMARY.md","content":"…"})
think  · turn 2

answer (2 turn(s)): I read all four files and wrote SUMMARY.md …

SUMMARY.md now exists on disk: …/sample-files/SUMMARY.md
```

### A side effect that outlives the process

`SUMMARY.md` is written into `sample-files/` and is **still there after the
program exits** — the first example that leaves something behind on disk. It is
`.gitignore`d so re-runs stay clean.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
```

## Run

```sh
cd examples/go/04-filesystem-agent
go run .
go run . --prompt "List the files and tell me which one mentions nutrients."
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint,
and `--model` overrides the id for a single run — same as 03.
