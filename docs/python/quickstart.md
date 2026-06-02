# Quickstart — Python

> 🚧 **Stub.** Full Python guide pages are in progress. The Python implementation tracks the same
> specification as Rust; until these pages fill in, read the language-agnostic
> [quickstart guide](../guides/quickstart.md) for the concepts and follow the
> [Rust quickstart](../rust/quickstart.md) for the shape of the code.

## The shape

The five-line path is the same in every language:

1. Construct a model interface (Ollama for zero-setup local runs, or a hosted provider).
2. Build a conversational harness from the model — the preset defaults every required component.
3. Create a task from an instruction.
4. Run it and await the result.
5. Match the result — `Success` carries the output, turn count, usage, and post-run session state.

See the [architecture concept](../concepts/architecture.md) for what those components are.

## Runnable examples

Three runnable Python programs build the path up step by step (`HarnessBuilder.conversational(model)`
+ `Task.simple(...)`):

- [`examples/python/01-hello-agent`](../../examples/python/01-hello-agent) — the smallest live agent: one turn, one greeting.
- [`examples/python/02-conversational-repl`](../../examples/python/02-conversational-repl) — a chat loop that remembers earlier turns by threading session state.
- [`examples/python/03-tool-use`](../../examples/python/03-tool-use) — a ReAct agent that calls local tools in a Think → Act → Observe loop.
