# spore-core documentation

Docs for **engineers building on spore-core** — people who want to build a harness, or
different kinds of harnesses, with the library. This is the *how do I use this* layer, not
internals documentation. (For the canonical language-agnostic specification of every
component, see [`harness-engineering-concepts.md`](./harness-engineering-concepts.md).)

spore-core ships in four languages — **Rust**, **TypeScript**, **Python**, **Go** — from a
single specification. The concept, guide, and reference pages below are language-agnostic and
contain no code. Each guide has a matching per-language page (e.g. `guides/quickstart.md` ↔
`rust/quickstart.md`) that carries the code examples for that language. The docs site renders
these as language tabs driven by one global selector.

> **Language parity.** Rust is the reference implementation and has the fullest examples.
> TypeScript, Python, and Go currently have quickstart stubs; their full guide pages fill in as
> language parity lands.

## Start here

- **New to spore-core?** Read [concepts/architecture](./concepts/architecture.md), then run
  the [quickstart](./guides/quickstart.md).
- **Want to add a capability?** [guides/building-a-tool](./guides/building-a-tool.md).
- **Building a chat / multi-turn app?** [guides/conversation](./guides/conversation.md).
- **Assembling your own harness from parts?** [guides/custom-harness](./guides/custom-harness.md).

## Concepts — *the ideas, no code*

| Page | What it covers |
|------|----------------|
| [architecture](./concepts/architecture.md) | `Agent = Model + Harness`; the components and how they wire together |
| [loop-strategies](./concepts/loop-strategies.md) | ReAct, PlanExecute, SelfVerifying, HillClimbing, Ralph |
| [tools](./concepts/tools.md) | The three ways to give an agent a tool, and when to use each |
| [memory](./concepts/memory.md) | The storage seam, `SessionState`, episodic vs semantic memory |
| [observability](./concepts/observability.md) | The trace outbox, OTLP, viewing traces |
| [multi-agent](./concepts/multi-agent.md) | Composition, agent-as-tool, sequential handoff |

## Guides — *narrative how-to, no code*

| Page | Per-language code |
|------|-------------------|
| [quickstart](./guides/quickstart.md) | [rust](./rust/quickstart.md) · [ts](./typescript/quickstart.md) · [py](./python/quickstart.md) · [go](./go/quickstart.md) |
| [building-a-tool](./guides/building-a-tool.md) | [rust](./rust/building-a-tool.md) |
| [conversation](./guides/conversation.md) | [rust](./rust/conversation.md) |
| [custom-harness](./guides/custom-harness.md) | [rust](./rust/custom-harness.md) |

## Reference — *the spec surface, no code*

| Page | What it covers |
|------|----------------|
| [harness-builder](./reference/harness-builder.md) | Every builder method and its default |
| [stream-events](./reference/stream-events.md) | The `StreamEvent` variants and their ordering |
| [storage-seam](./reference/storage-seam.md) | `StorageProvider` and the four domain stores |

## Examples

Runnable end-to-end programs live in [`examples/`](../examples), organized by language:
`examples/rust/01-hello-agent`, `examples/rust/02-conversational-repl`, and so on. Each guide's
*see it in action* section links to the matching example.
