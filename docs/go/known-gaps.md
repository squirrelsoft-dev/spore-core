# Known Go gaps

The approved Go parity scope covers `spore-core` core behavior and evaluation-facing seams. The following Rust subsystems are intentionally documented rather than created in Go.

## MCP (`spore-mcp`)

Rust contains an MCP client crate that maps MCP tools into native harness tools. Go does not currently include an equivalent MCP bridge. Use native Go `Tool` implementations or wrap MCP externally before registering tools.

## sqlite-vec store (`spore-sqlvec`)

Rust contains a sqlite-vec-backed `VectorStore` crate. Go provides the core vector-store interface and in-memory behavior, but not a sqlite-vec provider. Applications that need persistent vector storage should implement `vectorstore.VectorStore` against their chosen database.

## Petri microVM sandbox

Rust has a feature-gated Petri microVM sandbox. Go does not include that sandbox. Commands that request unavailable microVM behavior should be treated as unsupported by the configured sandbox, matching the bounded parity scope.

These are not regressions in the Go harness parity work; they are explicit out-of-scope subsystem gaps.
