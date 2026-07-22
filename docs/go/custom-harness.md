# Customizing the Go harness

`HarnessConfig` wires the runtime seams:

- `Agent`: model/agent implementation.
- `ToolRegistry` or `CatalogueRegistry`: tool catalogue and dispatcher.
- `Sandbox`: environment boundary for filesystem/process/network work.
- `ContextManager`: prompt/session management.
- `TerminationPolicy`: continuation/stop policy.
- `RunStore`: optional storage for run-local and durable tool state.
- `EscalationMode`: autonomous propagation vs pause-to-human behavior.

## Prompt chunks

`WithChunkProvider(provider)` lets harness sub-phases load prompt chunks. Because the root Go package cannot import `promptassembly` without a package cycle, this setter accepts a structural provider: any value with `Load(context.Context)` returning a slice/array of structs with `ID` and `Content` string fields plus an error.

SelfVerifying reads the chunk with ID `role-evaluator`; if missing or loading fails, it falls back to the built-in `RoleEvaluatorChunk`.

## Deferred tool exposure

`WithToolSearch(ToolSearchConfig{...})` wraps the effective registry with `ToolSearchRegistry`:

- only `CoreTools` and the `tool_search` meta-tool are advertised up front;
- `tool_search` searches non-core tool schemas by name/description;
- real tool calls still dispatch to the inner registry by name.

This is useful for large catalogues where advertising every schema would waste prompt budget.

## Embeddings and vectors

The Go port exposes core `embedding` and `vectorstore` packages:

- `embedding.EmbeddingModel` supports batch `Embed([]string)` and single-item `EmbedOne(string)`.
- `embedding.NewOllamaEmbeddingModel(modelID)` and `embedding.NewOpenAIEmbeddingModel(apiKey, modelID)` provide provider-backed implementations.
- `vectorstore.VectorStore` supports batch add, query, and filter-based delete with a deleted-count return.

The sqlite-vec-backed Rust crate is intentionally not ported; see `known-gaps.md`.
